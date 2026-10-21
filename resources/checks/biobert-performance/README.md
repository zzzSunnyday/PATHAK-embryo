# BioBERT 性能检查源码
这些源码来自本次独立实验。实际输入、编译 JAR、原始轮次、资源采样和未采用候选保存在项目的 .backups/biobert-performance-20261021T161556Z/。
源码中的工作目录以执行文件所在的新 .backups 子目录推导，不能直接在本源码目录中运行内部脚本。
复测入口是 resources/checks/benchmark_biobert_global8.py；它复制冻结输入和工具到新目录，并在独立 JVM 中使用 CPU2 基线及 global8/map2/OMP2 候选。GPU 代码仅供审计失败候选。
准备入口可使用 --prepare-only 检查；完整复测默认执行三轮纯模型/端到端比较、全库 batch1/8/16 和独立进程重载核查。它不修改正式模型、向量或加载代码。
当前正式结果见 [性能报告](../reports/BIOBERT_PERFORMANCE_REPORT.md)。复测入口从本目录复制适配当前路径的源码，冻结输入与 JAR 仍来自原审计目录；不修改原始审计记录。
