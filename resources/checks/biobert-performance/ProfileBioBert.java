package pathak.performance;

import com.johnsnowlabs.ml.tensorflow.*;
import com.johnsnowlabs.nlp.annotators.common.*;
import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.lang.reflect.Field;
import java.lang.management.ManagementFactory;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.tensorflow.*;
import org.tensorflow.proto.framework.*;
import scala.Tuple2;
import scala.collection.JavaConverters;
import scala.collection.Seq;

/** Experimental instrumentation only. Never rewrites a production artifact. */
public final class ProfileBioBert {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static long cpuTime() {
        return ((com.sun.management.OperatingSystemMXBean)ManagementFactory.getOperatingSystemMXBean()).getProcessCpuTime();
    }
    private static void floats(Path p,float[] values) throws Exception {
        ByteBuffer b=ByteBuffer.allocate(values.length*4).order(ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(values);Files.write(p,b.array());
    }
    public static Engine prepare(BertSentenceEmbeddings model,List<String> texts,int batchSize,String grouping) throws Exception {
        return new Engine(model,texts,batchSize,grouping);
    }
    public static String configuration(int intra,int inter,boolean privatePool) throws Exception {
        ConfigProto c=ConfigProto.newBuilder().setIntraOpParallelismThreads(intra).setInterOpParallelismThreads(inter).setUsePerSessionThreads(privatePool).build();
        List<Integer> bytes=new ArrayList<>();for(byte b:c.toByteArray())bytes.add((int)b&255);
        return JSON.writeValueAsString(bytes);
    }
    /** Profile the identical sentence FunctionDef through a direct call, B=1.
     * MapDefun does not propagate inner op stats on this TF build. This is an
     * explicitly labelled diagnostic copy, never a replacement serving graph.
     */
    public static String traceSentenceFunction(BertSentenceEmbeddings model,String text,String prefix) throws Exception {
        TensorflowBert source=model.getModelIfNotSet();
        GraphDef graph=GraphDef.parseFrom(source.tensorflowWrapper().graph());NodeDef map=null;
        for(NodeDef n:graph.getNodeList())if(n.getOp().equals("MapDefun"))map=n;
        if(map==null)throw new IllegalArgumentException("Expected the accepted MapDefun graph");
        GraphDef.Builder revised=graph.toBuilder().clearNode();
        for(NodeDef n:graph.getNodeList()) {
            if(n.getName().equals("pooled_output"))n=n.toBuilder().setInput(0,"pathak_profile_direct_call").build();
            revised.addNode(n);
        }
        List<String> inputs=new ArrayList<>();
        for(int i=0;i<3;i++) {
            String name="pathak_profile_squeeze_"+i;inputs.add(name);
            revised.addNode(NodeDef.newBuilder().setName(name).setOp("Squeeze").addInput(map.getInput(i))
                .putAttr("T",AttrValue.newBuilder().setType(DataType.DT_INT32).build())
                .putAttr("squeeze_dims",AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addI(0)).build()));
        }
        inputs.addAll(map.getInputList().subList(3,map.getInputCount()));
        AttrValue.ListValue.Builder tin=AttrValue.ListValue.newBuilder().addAllType(map.getAttrOrThrow("Targuments").getList().getTypeList())
                .addAllType(map.getAttrOrThrow("Tcaptured").getList().getTypeList());
        revised.addNode(NodeDef.newBuilder().setName("pathak_profile_direct_call").setOp("PartitionedCall").addAllInput(inputs)
            .putAttr("Tin",AttrValue.newBuilder().setList(tin).build())
            .putAttr("Tout",AttrValue.newBuilder().setList(AttrValue.ListValue.newBuilder().addType(DataType.DT_FLOAT)).build())
            .putAttr("f",map.getAttrOrThrow("f")));
        Field field=source.getClass().getDeclaredField("configProtoBytes");field.setAccessible(true);
        @SuppressWarnings("unchecked") scala.Option<byte[]> config=(scala.Option<byte[]>)field.get(source);
        TensorflowWrapper wrapper=new TensorflowWrapper(source.tensorflowWrapper().variables(),revised.build().toByteArray());
        Session session=wrapper.getTFSessionWithSignature(config,false,false,model.getSignatures());
        Seq<WordpieceTokenizedSentence> tokens=model.tokenize(Sentence.fromTexts(JavaConverters.asScalaBufferConverter(Arrays.asList(text)).asScala().toSeq()));
        List<Tuple2<WordpieceTokenizedSentence,Object>> selected=Arrays.asList(new Tuple2<>(tokens.apply(0),Integer.valueOf(0)));
        int[] ids=source.encode(JavaConverters.asScalaBufferConverter(selected).asScala().toSeq(),model.getMaxSentenceLength()).apply(0);
        Batch batch=new Batch(new int[][]{ids},Arrays.asList(0));
        try {
            for(int i=0;i<2;i++)for(Tensor t:session.runner().feed("input_ids",batch.ids).feed("input_mask",batch.mask).feed("segment_ids",batch.segments).fetch("pooled_output").run())t.close();
            Session.Run run=session.runner().feed("input_ids",batch.ids).feed("input_mask",batch.mask).feed("segment_ids",batch.segments).fetch("pooled_output")
                .setOptions(RunOptions.newBuilder().setTraceLevel(RunOptions.TraceLevel.FULL_TRACE).setOutputPartitionGraphs(true).build()).runAndFetchMetadata();
            try {
                Files.write(Paths.get(prefix+".pb"),run.metadata.toByteArray());
                floats(Paths.get(prefix+".bin"),TensorResources.extractFloats(run.outputs.get(0),scala.Option.empty()));
                StringBuilder out=new StringBuilder("device\tthread_id\tnode\tstart_us\top_us\tall_us\tlabel\n");Map<String,Object> devices=new LinkedHashMap<>();
                for(DeviceStepStats d:run.metadata.getStepStats().getDevStatsList()) {
                    devices.put(d.getDevice(),d.getNodeStatsCount());
                    for(NodeExecStats n:d.getNodeStatsList())out.append(d.getDevice()).append('\t').append(n.getThreadId()).append('\t').append(n.getNodeName()).append('\t')
                        .append(n.getAllStartMicros()).append('\t').append(n.getOpEndRelMicros()-n.getOpStartRelMicros()).append('\t').append(n.getAllEndRelMicros()).append('\t').append(n.getTimelineLabel().replace('\n',' ')).append('\n');
                }
                Files.write(Paths.get(prefix+".tsv"),out.toString().getBytes("UTF-8"));return JSON.writeValueAsString(devices);
            } finally {for(Tensor t:run.outputs)t.close();}
        } finally {batch.close();session.close();}
    }
    public static SparkMeter meter(org.apache.spark.SparkContext sc) {
        SparkMeter m=new SparkMeter();sc.addSparkListener(m);return m;
    }
    public static final class SparkMeter extends org.apache.spark.scheduler.SparkListener {
        private final List<Map<String,Object>> tasks=new ArrayList<>();
        public synchronized void reset(){tasks.clear();}
        @Override public synchronized void onTaskEnd(org.apache.spark.scheduler.SparkListenerTaskEnd event) {
            org.apache.spark.executor.TaskMetrics m=event.taskMetrics();if(m==null)return;
            Map<String,Object> r=new LinkedHashMap<>();r.put("stage",event.stageId());r.put("task",event.taskInfo().taskId());
            r.put("launch_ms",event.taskInfo().launchTime());r.put("finish_ms",event.taskInfo().finishTime());
            r.put("executor_run_ms",m.executorRunTime());r.put("executor_cpu_ns",m.executorCpuTime());
            r.put("deserialize_ms",m.executorDeserializeTime());r.put("serialize_ms",m.resultSerializationTime());
            r.put("jvm_gc_ms",m.jvmGCTime());r.put("result_bytes",m.resultSize());tasks.add(r);
        }
        public synchronized String report() throws Exception{return JSON.writeValueAsString(tasks);}
    }
    public static final class Batch implements AutoCloseable {
        final int[] positions;
        final TensorResources resources=new TensorResources();
        final Tensor ids,mask,segments;
        final int length;
        Batch(int[][] all,List<Integer> selected) {
            positions=selected.stream().mapToInt(Integer::intValue).toArray();
            int longest=0;for(int p:positions)longest=Math.max(longest,all[p].length);length=longest;
            int size=positions.length*length;
            int[] iv=new int[size],mv=new int[size],sv=new int[size];
            for(int row=0;row<positions.length;row++) {
                int[] source=all[positions[row]];
                System.arraycopy(source,0,iv,row*length,source.length);
                for(int j=0;j<source.length;j++)mv[row*length+j]=source[j]==0?0:1;
            }
            long[] shape=new long[]{positions.length,length};
            org.tensorflow.ndarray.buffer.IntDataBuffer ib=resources.createIntBuffer(size);ib.write(iv);
            org.tensorflow.ndarray.buffer.IntDataBuffer mb=resources.createIntBuffer(size);mb.write(mv);
            org.tensorflow.ndarray.buffer.IntDataBuffer sb=resources.createIntBuffer(size);sb.write(sv);
            ids=resources.createIntBufferTensor(shape,ib);mask=resources.createIntBufferTensor(shape,mb);segments=resources.createIntBufferTensor(shape,sb);
        }
        public void close(){resources.clearTensors();}
    }
    public static final class Engine implements AutoCloseable {
        private final Session session;
        private final List<Batch> batches=new ArrayList<>();
        private final int[][] ids;
        private final int count;
        private final Map<String,Object> plan=new LinkedHashMap<>();
        Engine(BertSentenceEmbeddings model,List<String> texts,int batchSize,String grouping) throws Exception {
            long start=System.nanoTime();
            count=texts.size();ids=new int[count][];
            TensorflowBert backend=model.getModelIfNotSet();
            Field field=backend.getClass().getDeclaredField("configProtoBytes");field.setAccessible(true);
            @SuppressWarnings("unchecked") scala.Option<byte[]> config=(scala.Option<byte[]>)field.get(backend);
            session=backend.tensorflowWrapper().getTFSessionWithSignature(config,false,false,model.getSignatures());
            long t=System.nanoTime();
            Seq<WordpieceTokenizedSentence> tokens=model.tokenize(Sentence.fromTexts(JavaConverters.asScalaBufferConverter(texts).asScala().toSeq()));
            double tokenize=(System.nanoTime()-t)/1e9;t=System.nanoTime();
            // Encode each row once with the exact production tokenizer and truncation.
            for(int i=0;i<count;i++) {
                List<Tuple2<WordpieceTokenizedSentence,Object>> row=Arrays.asList(new Tuple2<>(tokens.apply(i),Integer.valueOf(i)));
                ids[i]=backend.encode(JavaConverters.asScalaBufferConverter(row).asScala().toSeq(),model.getMaxSentenceLength()).apply(0);
            }
            double encode=(System.nanoTime()-t)/1e9;t=System.nanoTime();
            List<Integer> order=new ArrayList<>();for(int i=0;i<count;i++)order.add(i);
            if(!grouping.equals("input"))order.sort(Comparator.comparingInt(i->ids[i].length));
            for(int i=0;i<count;) {
                int end=Math.min(count,i+batchSize);
                if(grouping.equals("exact")) {
                    int same=i+1;while(same<end&&ids[order.get(same)].length==ids[order.get(i)].length)same++;
                    end=same;
                }
                batches.add(new Batch(ids,order.subList(i,end)));i=end;
            }
            long useful=0,padded=0;int max=0;
            for(int[] row:ids){useful+=row.length;max=Math.max(max,row.length);}
            for(Batch b:batches)padded+=(long)b.length*b.positions.length;
            plan.put("rows",count);plan.put("batch_size",batchSize);plan.put("grouping",grouping);
            plan.put("batch_count",batches.size());plan.put("effective_tokens",useful);plan.put("padded_tokens",padded);
            plan.put("max_token_length",max);plan.put("tokenize_seconds",tokenize);plan.put("encode_ids_seconds",encode);
            plan.put("tensor_prepare_seconds",(System.nanoTime()-t)/1e9);plan.put("prepare_seconds",(System.nanoTime()-start)/1e9);
            plan.put("tensorflow_version",TensorFlow.version());plan.put("config_proto",config.isDefined()?ConfigProto.parseFrom(config.get()).toString():"default");
        }
        public String describe() throws Exception {return JSON.writeValueAsString(plan);}
        public String lengths() throws Exception {
            int[] lengths=new int[count];for(int i=0;i<count;i++)lengths[i]=ids[i].length;
            return JSON.writeValueAsString(lengths);
        }
        public String inputs() throws Exception{return JSON.writeValueAsString(ids);}
        public void exportInputs(String filename) throws Exception {
            Map<String,Object> out=new LinkedHashMap<>();out.put("input_ids",ids);out.put("plan",plan);
            Files.write(Paths.get(filename),JSON.writeValueAsBytes(out));
        }
        private Session.Runner runner(Batch b) {
            return session.runner().feed("input_ids",b.ids).feed("input_mask",b.mask).feed("segment_ids",b.segments).fetch("pooled_output");
        }
        public String run(String outputFilename) throws Exception {
            return runConcurrent(outputFilename,1);
        }
        public String runConcurrent(String outputFilename,int concurrentCalls) throws Exception {
            float[] result=new float[count*768];long totalStart=System.nanoTime(),cpuStart=cpuTime();
            AtomicLong runs=new AtomicLong(),copies=new AtomicLong(),closes=new AtomicLong();
            ExecutorService executor=concurrentCalls>1?Executors.newFixedThreadPool(concurrentCalls):null;
            List<Callable<Void>> jobs=new ArrayList<>();
            for(Batch b:batches)jobs.add(()->{
                Session.Runner r=runner(b);long t=System.nanoTime();List<Tensor> outputs=r.run();runs.addAndGet(System.nanoTime()-t);
                try {
                    t=System.nanoTime();float[] values=TensorResources.extractFloats(outputs.get(0),scala.Option.empty());
                    if(values.length!=b.positions.length*768)throw new IllegalStateException("Unexpected output shape");
                    for(int i=0;i<b.positions.length;i++)System.arraycopy(values,i*768,result,b.positions[i]*768,768);
                    copies.addAndGet(System.nanoTime()-t);
                } finally {long c=System.nanoTime();for(Tensor v:outputs)v.close();closes.addAndGet(System.nanoTime()-c);}
                return null;
            });
            try {
                if(executor==null){for(Callable<Void> job:jobs)job.call();}
                else {for(Future<Void> f:executor.invokeAll(jobs))f.get();}
            } finally {if(executor!=null)executor.shutdown();}
            double wall=(System.nanoTime()-totalStart)/1e9,cpu=(cpuTime()-cpuStart)/1e9;
            Map<String,Object> out=new LinkedHashMap<>(plan);
            out.put("wall_seconds",wall);out.put("tf_runner_seconds_summed",runs.get()/1e9);out.put("output_copy_seconds_summed",copies.get()/1e9);
            out.put("output_close_seconds_summed",closes.get()/1e9);out.put("concurrent_calls",concurrentCalls);out.put("process_cpu_seconds",cpu);out.put("average_cpu_cores",cpu/wall);out.put("rows_per_second",count/wall);
            if(!outputFilename.isEmpty())floats(Paths.get(outputFilename),result);
            return JSON.writeValueAsString(out);
        }
        public String trace(String prefix) throws Exception {
            Batch b=batches.get(0);
            Session.Run run=runner(b).setOptions(RunOptions.newBuilder().setTraceLevel(RunOptions.TraceLevel.FULL_TRACE).setOutputPartitionGraphs(true).build()).runAndFetchMetadata();
            try {
                Files.write(Paths.get(prefix+".pb"),run.metadata.toByteArray());
                StringBuilder text=new StringBuilder("device\tthread_id\tnode\tstart_us\top_us\tall_us\tscheduled_us\tlabel\n");
                Map<String,Object> devices=new LinkedHashMap<>();
                for(DeviceStepStats d:run.metadata.getStepStats().getDevStatsList()) {
                    devices.put(d.getDevice(),d.getNodeStatsCount());
                    for(NodeExecStats n:d.getNodeStatsList())text.append(d.getDevice()).append('\t').append(n.getThreadId()).append('\t').append(n.getNodeName()).append('\t')
                        .append(n.getAllStartMicros()).append('\t').append(n.getOpEndRelMicros()-n.getOpStartRelMicros()).append('\t').append(n.getAllEndRelMicros())
                        .append('\t').append(n.getScheduledMicros()).append('\t').append(n.getTimelineLabel().replace('\n',' ')).append('\n');
                }
                Files.write(Paths.get(prefix+".tsv"),text.toString().getBytes("UTF-8"));
                int i=0;for(GraphDef g:run.metadata.getPartitionGraphsList())Files.write(Paths.get(prefix+"_graph"+(i++)+".pb"),g.toByteArray());
                return JSON.writeValueAsString(devices);
            } finally {for(Tensor t:run.outputs)t.close();}
        }
        public void close(){for(Batch b:batches)b.close();}
    }
}
