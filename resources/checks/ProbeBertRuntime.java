package pathak.audit;

import com.johnsnowlabs.ml.tensorflow.*;
import com.johnsnowlabs.nlp.annotators.common.*;
import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.lang.reflect.Field;
import java.util.zip.*;
import org.tensorflow.*;
import org.tensorflow.proto.framework.*;
import scala.Tuple2;
import scala.collection.JavaConverters;
import scala.collection.Seq;

/** Read-only runtime experiment. Production model files are never rewritten. */
public final class ProbeBertRuntime {
    private static void writeFloats(Path file, float[] values) throws Exception {
        ByteBuffer buffer=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().put(values);Files.write(file,buffer.array());
    }

    private static String nodeLines(GraphDef graph) {
        StringBuilder text=new StringBuilder();
        for(NodeDef n:graph.getNodeList())text.append(n.getName()).append('\t').append(n.getOp())
                .append('\t').append(String.join(",",n.getInputList())).append('\n');
        return text.toString();
    }

    public static void dumpGraph(String archive,String output) throws Exception {
        try(ZipFile zip=new ZipFile(archive)) {
            GraphDef graph=GraphDef.parseFrom(zip.getInputStream(zip.getEntry("saved_model.pb")));
            Files.write(Paths.get(output),nodeLines(graph).getBytes("UTF-8"));
        }
    }

    private static float[] predict(Session session,Seq<int[]> encoded,Path profile) throws Exception {
        TensorResources resources=new TensorResources();
        int rows=encoded.size(),length=encoded.apply(0).length;
        int[] ids=new int[rows*length],mask=new int[ids.length],segments=new int[ids.length];
        for(int i=0;i<rows;i++)for(int j=0;j<length;j++) {
            int value=encoded.apply(i)[j];ids[i*length+j]=value;mask[i*length+j]=value==0?0:1;
        }
        long[] shape=new long[]{rows,length};
        org.tensorflow.ndarray.buffer.IntDataBuffer ib=resources.createIntBuffer(ids.length);ib.write(ids);
        org.tensorflow.ndarray.buffer.IntDataBuffer mb=resources.createIntBuffer(mask.length);mb.write(mask);
        org.tensorflow.ndarray.buffer.IntDataBuffer sb=resources.createIntBuffer(segments.length);sb.write(segments);
        Tensor it=resources.createIntBufferTensor(shape,ib),mt=resources.createIntBufferTensor(shape,mb),st=resources.createIntBufferTensor(shape,sb);
        Session.Runner runner=session.runner().feed("input_ids",it).feed("input_mask",mt).feed("segment_ids",st).fetch("pooled_output");
        List<Tensor> outputs;
        if(profile!=null) {
            Session.Run run=runner.setOptions(RunOptions.newBuilder().setOutputPartitionGraphs(true).build()).runAndFetchMetadata();
            outputs=run.outputs;
            int part=0;
            for(GraphDef graph:run.metadata.getPartitionGraphsList()) {
                Files.write(Paths.get(profile.toString()+"_"+part+".tsv"),nodeLines(graph).getBytes("UTF-8"));part++;
                Files.write(Paths.get(profile.toString()+"_"+(part-1)+".pb"),graph.toByteArray());
            }
        } else outputs=runner.run();
        try{return TensorResources.extractFloats(outputs.get(0),scala.Option.empty());}
        finally {for(Tensor tensor:outputs)tensor.close();resources.clearTensors();}
    }

    private static Seq<int[]> encode(TensorflowBert backend,Seq<WordpieceTokenizedSentence> tokens,List<Integer> rows,int limit) {
        List<Tuple2<WordpieceTokenizedSentence,Object>> selected=new ArrayList<>();
        for(int index:rows)selected.add(new Tuple2<>(tokens.apply(index),Integer.valueOf(index)));
        return backend.encode(JavaConverters.asScalaBufferConverter(selected).asScala().toSeq(),limit);
    }

    public static String probe(BertSentenceEmbeddings model,List<String> texts,String mode,String outputDirectory) throws Exception {
        Path output=Paths.get(outputDirectory);Files.createDirectories(output);
        TensorflowBert source=model.getModelIfNotSet();
        StringBuilder identity=new StringBuilder("TensorFlow "+TensorFlow.version()+"\n");
        for(Field field:source.getClass().getDeclaredFields())if(field.getName().contains("configProto")) {
            field.setAccessible(true);Object value=field.get(source);identity.append(field.getName()).append("=").append(value).append('\n');
        }
        ConfigProto.Builder configuration=ConfigProto.newBuilder().setIntraOpParallelismThreads(2).setInterOpParallelismThreads(2);
        if(mode.equals("threads1"))configuration.setIntraOpParallelismThreads(1).setInterOpParallelismThreads(1);
        if(mode.equals("remap_off"))configuration.setGraphOptions(GraphOptions.newBuilder().setRewriteOptions(
                RewriterConfig.newBuilder().setRemapping(RewriterConfig.Toggle.OFF)));
        if(mode.equals("meta_off"))configuration.setGraphOptions(GraphOptions.newBuilder().setRewriteOptions(
                RewriterConfig.newBuilder().setDisableMetaOptimizer(true)));
        scala.Option<byte[]> config=mode.equals("default")?scala.Option.empty():scala.Option.apply(configuration.build().toByteArray());
        identity.append("experiment=").append(mode).append('\n').append(configuration.build().toString());
        Files.write(output.resolve("session.txt"),identity.toString().getBytes("UTF-8"));
        TensorflowWrapper wrapper=new TensorflowWrapper(source.tensorflowWrapper().variables(),source.tensorflowWrapper().graph());
        Session session=wrapper.getTFSessionWithSignature(config,false,false,model.getSignatures());
        TensorflowBert backend=new TensorflowBert(wrapper,model.sentenceStartTokenId(),model.sentenceEndTokenId(),config,model.getSignatures());
        Seq<WordpieceTokenizedSentence> tokens=model.tokenize(Sentence.fromTexts(JavaConverters.asScalaBufferConverter(texts).asScala().toSeq()));
        List<Integer> indices=new ArrayList<>();
        for(int i=0;i<texts.size();i++)indices.add(i);
        long start=System.nanoTime();
        try {
            float[] singles=new float[texts.size()*768];
            for(int i=0;i<texts.size();i++) {
                float[] vector=predict(session,encode(backend,tokens,Arrays.asList(i),model.getMaxSentenceLength()),i==0?output.resolve("single_optimized"):null);
                System.arraycopy(vector,0,singles,i*768,768);
            }
            writeFloats(output.resolve("singles.bin"),singles);
            writeFloats(output.resolve("mixed.bin"),predict(session,encode(backend,tokens,indices,model.getMaxSentenceLength()),output.resolve("mixed_optimized")));
            List<Integer> repeats=new ArrayList<>();for(int i=0;i<8;i++)repeats.add(0);
            writeFloats(output.resolve("repeated8.bin"),predict(session,encode(backend,tokens,repeats,model.getMaxSentenceLength()),null));
        } finally {session.close();}
        return mode+" "+String.format(java.util.Locale.ROOT,"%.3f",(System.nanoTime()-start)/1e9)+"s "+identity.toString().replace('\n',' ');
    }

    private static void layerRun(Session session,Seq<int[]> encoded,List<String> names,Path folder) throws Exception {
        Files.createDirectories(folder);TensorResources resources=new TensorResources();
        int rows=encoded.size(),length=encoded.apply(0).length;
        int[] ids=new int[rows*length],mask=new int[ids.length],segments=new int[ids.length];
        for(int i=0;i<rows;i++)for(int j=0;j<length;j++) {
            ids[i*length+j]=encoded.apply(i)[j];mask[i*length+j]=ids[i*length+j]==0?0:1;
        }
        long[] shape=new long[]{rows,length};
        org.tensorflow.ndarray.buffer.IntDataBuffer ib=resources.createIntBuffer(ids.length);ib.write(ids);
        org.tensorflow.ndarray.buffer.IntDataBuffer mb=resources.createIntBuffer(mask.length);mb.write(mask);
        org.tensorflow.ndarray.buffer.IntDataBuffer sb=resources.createIntBuffer(segments.length);sb.write(segments);
        Session.Runner runner=session.runner().feed("input_ids",resources.createIntBufferTensor(shape,ib))
                .feed("input_mask",resources.createIntBufferTensor(shape,mb))
                .feed("segment_ids",resources.createIntBufferTensor(shape,sb));
        for(String name:names)runner.fetch(name);
        List<Tensor> outputs=runner.run();StringBuilder index=new StringBuilder();
        try {
            for(int i=0;i<outputs.size();i++) {
                Tensor tensor=outputs.get(i);String filename=String.format("%03d.bin",i);
                writeFloats(folder.resolve(filename),TensorResources.extractFloats(tensor,scala.Option.empty()));
                index.append(filename).append('\t').append(names.get(i)).append('\t')
                        .append(Arrays.toString(tensor.shape().asArray())).append('\n');
            }
            Files.write(folder.resolve("index.tsv"),index.toString().getBytes("UTF-8"));
            Files.write(folder.resolve("input_shape.txt"),(rows+" "+length).getBytes("UTF-8"));
        } finally {for(Tensor tensor:outputs)tensor.close();resources.clearTensors();}
    }

    public static void layers(BertSentenceEmbeddings model,List<String> texts,String outputDirectory) throws Exception {
        List<String> names=new ArrayList<>();
        names.add("bert/embeddings/add");names.add("bert/embeddings/add_1");
        for(String sub:new String[]{"moments/mean","moments/variance","batchnorm/add","batchnorm/Rsqrt","batchnorm/mul","batchnorm/mul_1","batchnorm/mul_2","batchnorm/sub","batchnorm/add_1"})
            names.add("bert/embeddings/LayerNorm/"+sub);
        for(String sub:new String[]{"query","key","value"})names.add("bert/encoder/layer_0/attention/self/"+sub+"/MatMul");
        for(int i=0;i<12;i++) {
            String prefix="bert/encoder/layer_"+i+"/";
            names.add(prefix+"attention/output/LayerNorm/batchnorm/add_1");
            names.add(prefix+"intermediate/dense/BiasAdd");names.add(prefix+"intermediate/dense/Tanh");
            names.add(prefix+"intermediate/dense/mul_3");names.add(prefix+"output/LayerNorm/batchnorm/add_1");
        }
        names.add("pooled_output");
        TensorflowBert source=model.getModelIfNotSet();
        TensorflowWrapper wrapper=new TensorflowWrapper(source.tensorflowWrapper().variables(),source.tensorflowWrapper().graph());
        ConfigProto config=ConfigProto.newBuilder().setIntraOpParallelismThreads(2).setInterOpParallelismThreads(2)
                .setGraphOptions(GraphOptions.newBuilder().setRewriteOptions(RewriterConfig.newBuilder().setRemapping(RewriterConfig.Toggle.OFF))).build();
        scala.Option<byte[]> option=scala.Option.apply(config.toByteArray());
        Session session=wrapper.getTFSessionWithSignature(option,false,false,model.getSignatures());
        TensorflowBert backend=new TensorflowBert(wrapper,model.sentenceStartTokenId(),model.sentenceEndTokenId(),option,model.getSignatures());
        Seq<WordpieceTokenizedSentence> tokens=model.tokenize(Sentence.fromTexts(JavaConverters.asScalaBufferConverter(texts).asScala().toSeq()));
        List<Integer> mixed=new ArrayList<>(),repeated=new ArrayList<>();
        for(int i=0;i<texts.size();i++)mixed.add(i);
        for(int i=0;i<8;i++)repeated.add(0);
        Path output=Paths.get(outputDirectory);
        try {
            layerRun(session,encode(backend,tokens,Arrays.asList(0),128),names,output.resolve("single"));
            layerRun(session,encode(backend,tokens,repeated,128),names,output.resolve("repeated8"));
            layerRun(session,encode(backend,tokens,mixed,128),names,output.resolve("mixed8"));
        } finally {session.close();}
    }
}
