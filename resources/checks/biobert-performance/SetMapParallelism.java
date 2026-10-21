import java.nio.file.*;
import org.tensorflow.proto.framework.*;
public final class SetMapParallelism {
    public static void main(String[] args)throws Exception{
        GraphDef g=GraphDef.parseFrom(Files.readAllBytes(Paths.get(args[0])));
        GraphDef.Builder b=g.toBuilder().clearNode();int changed=0;
        for(NodeDef n:g.getNodeList()){
            if(n.getOp().equals("MapDefun")){b.addNode(n.toBuilder().putAttr("max_intra_op_parallelism",AttrValue.newBuilder().setI(Integer.parseInt(args[2])).build()));changed++;}
            else b.addNode(n);
        }
        if(changed!=1)throw new IllegalStateException("Expected one graph-native sentence map");
        Files.write(Paths.get(args[1]),b.build().toByteArray());
    }
}
