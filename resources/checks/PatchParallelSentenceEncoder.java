import java.nio.file.*;
import java.util.*;
import org.tensorflow.*;
import org.tensorflow.proto.framework.*;

/** Graph-native parallel map. Each invocation preserves legacy sentence arithmetic. */
public final class PatchParallelSentenceEncoder {
    static AttrValue type(DataType t){return AttrValue.newBuilder().setType(t).build();}
    static AttrValue bool(boolean b){return AttrValue.newBuilder().setB(b).build();}
    static AttrValue types(List<DataType> values){return AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addAllType(values)).build();}
    static NodeDef.Builder node(String n,String op,String... inputs){NodeDef.Builder b=NodeDef.newBuilder().setName(n).setOp(op);for(String x:inputs)b.addInput(x);return b;}
    static NodeDef iconst(String name,long[] shape,int... values){
        TensorShapeProto.Builder s=TensorShapeProto.newBuilder();for(long v:shape)s.addDim(TensorShapeProto.Dim.newBuilder().setSize(v));
        TensorProto.Builder t=TensorProto.newBuilder().setDtype(DataType.DT_INT32).setTensorShape(s);for(int v:values)t.addIntVal(v);
        return node(name,"Const").putAttr("dtype",type(DataType.DT_INT32)).putAttr("value",AttrValue.newBuilder().setTensor(t).build()).build();
    }
    public static void main(String[] args)throws Exception{
        if(args.length<3)throw new IllegalArgumentException("Pass masked GraphDef, output GraphDef, and captured legacy runtime GraphDef");
        GraphDef source=GraphDef.parseFrom(Files.readAllBytes(Paths.get(args[0])));
        Map<String,NodeDef> nodes=new HashMap<>();for(NodeDef n:source.getNodeList())nodes.put(n.getName(),n);
        LinkedHashMap<String,String> captures=new LinkedHashMap<>();
        for(NodeDef n:source.getNodeList())if(n.getOp().equals("Identity")&&n.getName().endsWith("/read")&&nodes.get(n.getInput(0)).getOp().equals("VariableV2"))
            captures.put(String.format("weight_%03d",captures.size()),n.getName());
        GraphDef execution=args.length>2?GraphDef.parseFrom(Files.readAllBytes(Paths.get(args[2]))):source;
        Map<String,String> arguments=new HashMap<>();String[] fields={"input_ids","input_mask","segment_ids"};
        for(NodeDef n:execution.getNodeList())if(n.getOp().equals("_Arg"))arguments.put(n.getName(),fields[(int)n.getAttrOrThrow("index").getI()]);
        Set<String> available=new HashSet<>();for(NodeDef n:execution.getNodeList())available.add(n.getName());
        captures.entrySet().removeIf(entry->!available.contains(entry.getValue()));
        GraphDef.Builder body=execution.toBuilder().clearNode();
        for(NodeDef n:execution.getNodeList()){
            if(Arrays.asList(fields).contains(n.getName()) || n.getOp().equals("_Arg") || n.getOp().equals("_Retval"))continue;
            NodeDef.Builder revised=n.toBuilder().clearDevice();
            for(int i=0;i<n.getInputCount();i++){
                String value=n.getInput(i),bare=value.startsWith("^")?value.substring(1):value;
                String replacement=arguments.get(bare);
                if(replacement!=null)revised.setInput(i,value.startsWith("^")?"^"+replacement:replacement);
            }
            body.addNode(revised);
        }
        for(String field:fields)body.addNode(node("pathak_row_"+field,"Placeholder").putAttr("dtype",type(DataType.DT_INT32))
            .putAttr("shape",AttrValue.newBuilder().setShape(TensorShapeProto.newBuilder().addDim(TensorShapeProto.Dim.newBuilder().setSize(-1))).build()));
        body.addNode(iconst("pathak_axis",new long[]{},0));body.addNode(iconst("pathak_begin",new long[]{1},0));
        // CLS mask is 1 at runtime. Keep the batch dimension dynamic, as in the original graph.
        body.addNode(node("pathak_one","GatherV2","pathak_row_input_mask","pathak_axis","pathak_axis").putAttr("Tparams",type(DataType.DT_INT32))
            .putAttr("Tindices",type(DataType.DT_INT32)).putAttr("Taxis",type(DataType.DT_INT32)).putAttr("batch_dims",AttrValue.newBuilder().setI(0).build()));
        body.addNode(node("pathak_length","Sum","pathak_row_input_mask","pathak_axis").putAttr("T",type(DataType.DT_INT32)).putAttr("Tidx",type(DataType.DT_INT32)).putAttr("keep_dims",bool(false)));
        body.addNode(node("pathak_trim_size","Pack","pathak_length").putAttr("N",AttrValue.newBuilder().setI(1).build()).putAttr("T",type(DataType.DT_INT32)).putAttr("axis",AttrValue.newBuilder().setI(0).build()));
        body.addNode(node("pathak_encoder_shape","Pack","pathak_one","pathak_length").putAttr("N",AttrValue.newBuilder().setI(2).build()).putAttr("T",type(DataType.DT_INT32)).putAttr("axis",AttrValue.newBuilder().setI(0).build()));
        for(String field:fields){
            body.addNode(node("pathak_trim_"+field,"Slice","pathak_row_"+field,"pathak_begin","pathak_trim_size").putAttr("T",type(DataType.DT_INT32)).putAttr("Index",type(DataType.DT_INT32)));
            body.addNode(node(field,"Reshape","pathak_trim_"+field,"pathak_encoder_shape").putAttr("T",type(DataType.DT_INT32)).putAttr("Tshape",type(DataType.DT_INT32)));
        }
        body.addNode(node("pathak_row_embedding","Squeeze","pooled_output").putAttr("T",type(DataType.DT_FLOAT))
            .putAttr("squeeze_dims",AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addI(0)).build()));
        FunctionDef function;
        try(Graph graph=new Graph()){
            graph.importGraphDef(body.build());
            Signature.Builder signature=Signature.builder("PathakParallelSentenceEncoder");
            for(int i=0;i<fields.length;i++)signature.input("arg_"+i,graph.outputOrThrow("pathak_row_"+fields[i]+":0"));
            for(Map.Entry<String,String> entry:captures.entrySet())signature.input(entry.getKey(),graph.outputOrThrow(entry.getValue()+":0"));
            signature.output("embedding",graph.outputOrThrow("pathak_row_embedding:0"));
            try(ConcreteFunction f=ConcreteFunction.create(signature.build(),graph)){function=f.getFunctionDef();}
        }
        System.out.println("Function "+function.getSignature().getName()+", "+function.getSignature().getInputArgCount()+" args; "+function.getNodeDefCount()+" nodes");
        List<DataType> capturedTypes=new ArrayList<>();List<String> capturedNames=new ArrayList<>();
        for(int i=0;i<function.getSignature().getInputArgCount();i++){
            OpDef.ArgDef arg=function.getSignature().getInputArg(i);
            if(i<3){if(!arg.getName().equals("pathak_row_"+fields[i]))throw new IllegalStateException("Unexpected argument order: "+arg.getName());}
            else{
                String name=null;for(String candidate:captures.values())if(candidate.replace('/','_').toLowerCase(Locale.ROOT).equals(arg.getName()))name=candidate;
                if(name==null)throw new IllegalStateException("Unknown capture "+arg.getName());
                capturedNames.add(name);capturedTypes.add(arg.getType());
            }
        }
        GraphDef.Builder result=source.toBuilder().clearNode();result.getLibraryBuilder().addFunction(function);
        for(NodeDef n:source.getNodeList())if(n.getName().equals("pooled_output"))result.addNode(n.toBuilder().setInput(0,"pathak_parallel_sentence_encoder"));else result.addNode(n);
        NodeDef.Builder map=node("pathak_parallel_sentence_encoder","MapDefun",fields);
        map.addAllInput(capturedNames).putAttr("Targuments",types(Arrays.asList(DataType.DT_INT32,DataType.DT_INT32,DataType.DT_INT32)))
            .putAttr("Tcaptured",types(capturedTypes)).putAttr("output_types",types(Arrays.asList(DataType.DT_FLOAT)))
            .putAttr("output_shapes",AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addShape(TensorShapeProto.newBuilder().addDim(TensorShapeProto.Dim.newBuilder().setSize(768)))).build())
            .putAttr("f",AttrValue.newBuilder().setFunc(NameAttrList.newBuilder().setName(function.getSignature().getName())).build())
            .putAttr("max_intra_op_parallelism",AttrValue.newBuilder().setI(args.length>3?Integer.parseInt(args[3]):2).build());
        result.addNode(map);Files.write(Paths.get(args[1]),result.build().toByteArray());
        System.out.println("Written parallel graph; "+captures.size()+" shared, unchanged weight tensors");
    }
}
