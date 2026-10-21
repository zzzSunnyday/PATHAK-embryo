import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import org.tensorflow.proto.framework.AttrValue;
import org.tensorflow.proto.framework.DataType;
import org.tensorflow.proto.framework.GraphDef;
import org.tensorflow.proto.framework.NodeDef;
import org.tensorflow.proto.framework.TensorProto;
import org.tensorflow.proto.framework.TensorShapeProto;

/** Patch only the sentence readout, using the protobuf classes already in Spark NLP. */
public final class MaskedMeanGraph {
    private static final String PREFIX = "pathak_masked_mean/";

    private static AttrValue type(DataType dtype) {
        return AttrValue.newBuilder().setType(dtype).build();
    }

    private static NodeDef scalar(String name, int value) {
        TensorProto tensor = TensorProto.newBuilder().setDtype(DataType.DT_INT32)
                .setTensorShape(TensorShapeProto.newBuilder().build()).addIntVal(value).build();
        return NodeDef.newBuilder().setName(PREFIX + name).setOp("Const")
                .putAttr("dtype", type(DataType.DT_INT32))
                .putAttr("value", AttrValue.newBuilder().setTensor(tensor).build()).build();
    }

    private static NodeDef sum(String name, String input) {
        return NodeDef.newBuilder().setName(PREFIX + name).setOp("Sum")
                .addInput(input).addInput(PREFIX + "sequence_axis")
                .putAttr("T", type(DataType.DT_FLOAT))
                .putAttr("Tidx", type(DataType.DT_INT32))
                .putAttr("keep_dims", AttrValue.newBuilder().setB(false).build()).build();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("original.pb patched.pb");
        GraphDef graph = GraphDef.parseFrom(Files.readAllBytes(Paths.get(args[0])));
        Map<String, NodeDef> nodes = new LinkedHashMap<>();
        for (NodeDef node : graph.getNodeList()) {
            if (nodes.put(node.getName(), node) != null)
                throw new IllegalStateException("Duplicate node: " + node.getName());
            if (node.getName().startsWith(PREFIX))
                throw new IllegalStateException("Graph was already patched");
        }
        NodeDef pooled = nodes.get("pooled_output");
        if (pooled == null || !pooled.getOp().equals("Identity") || pooled.getInputCount() != 1)
            throw new IllegalStateException("Unexpected pooled_output");
        String meanName = pooled.getInput(0).replaceFirst(":0$", "");
        NodeDef mean = nodes.get(meanName);
        if (mean == null || !mean.getOp().equals("Mean") || mean.getInputCount() != 2)
            throw new IllegalStateException("Expected an unmasked Mean before pooled_output");
        String hidden = mean.getInput(0);
        if (!hidden.replaceFirst(":0$", "").equals("bert/encoder/Reshape_13"))
            throw new IllegalStateException("Unexpected hidden-state tensor: " + hidden);
        NodeDef mask = nodes.get("input_mask");
        if (mask == null || !mask.getOp().equals("Placeholder") ||
                mask.getAttrOrThrow("dtype").getType() != DataType.DT_INT32)
            throw new IllegalStateException("Expected int32 input_mask");
        System.out.println("ORIGINAL_POOLED\n" + pooled);
        System.out.println("ORIGINAL_MEAN\n" + mean);
        System.out.println("ORIGINAL_AXIS\n" + nodes.get(mean.getInput(1).replaceFirst(":0$", "")));
        GraphDef.Builder revised = graph.toBuilder().clearNode();
        for (NodeDef node : graph.getNodeList()) {
            revised.addNode(node.getName().equals("pooled_output")
                    ? node.toBuilder().setInput(0, PREFIX + "pooled").build() : node);
        }
        revised.addNode(scalar("sequence_axis", 1));
        revised.addNode(scalar("feature_axis", 2));
        revised.addNode(NodeDef.newBuilder().setName(PREFIX + "mask_float").setOp("Cast")
                .addInput("input_mask").putAttr("SrcT", type(DataType.DT_INT32))
                .putAttr("DstT", type(DataType.DT_FLOAT))
                .putAttr("Truncate", AttrValue.newBuilder().setB(false).build()));
        revised.addNode(NodeDef.newBuilder().setName(PREFIX + "mask").setOp("ExpandDims")
                .addInput(PREFIX + "mask_float").addInput(PREFIX + "feature_axis")
                .putAttr("T", type(DataType.DT_FLOAT)).putAttr("Tdim", type(DataType.DT_INT32)));
        revised.addNode(NodeDef.newBuilder().setName(PREFIX + "weighted_hidden").setOp("Mul")
                .addInput(hidden).addInput(PREFIX + "mask").putAttr("T", type(DataType.DT_FLOAT)));
        revised.addNode(sum("numerator", PREFIX + "weighted_hidden"));
        revised.addNode(sum("denominator", PREFIX + "mask"));
        revised.addNode(NodeDef.newBuilder().setName(PREFIX + "pooled").setOp("RealDiv")
                .addInput(PREFIX + "numerator").addInput(PREFIX + "denominator")
                .putAttr("T", type(DataType.DT_FLOAT)));
        GraphDef patched = revised.build();
        if (patched.getNodeCount() != graph.getNodeCount() + 8)
            throw new IllegalStateException("Unexpected node count");
        for (int i = 0; i < graph.getNodeCount(); i++) {
            NodeDef before = graph.getNode(i), after = patched.getNode(i);
            if (!before.getName().equals("pooled_output") && !before.equals(after))
                throw new IllegalStateException("Unexpected modification: " + before.getName());
        }
        Files.write(Paths.get(args[1]), patched.toByteArray());
        System.out.println("PATCHED: " + graph.getNodeCount() + " -> " + patched.getNodeCount() + " nodes");
        System.out.println("Only pooled_output input changed; all original weights/other nodes preserved.");
        System.out.println("Pooling: sum(hidden * cast(input_mask)[...,None],1) / sum(mask,1).");
    }
}
