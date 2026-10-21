# 上游来源与原始教学

上游项目：[nilesh-iiita/PATHAK](https://github.com/nilesh-iiita/PATHAK)。
本项目以其文献检索和 gene–GO 候选生成思路为起点，新增小鼠胚胎研究流程。
原作者将 PATHAK 应用于 Arabidopsis thaliana；本项目的小鼠研究结果需单独验证。

## 保留的原始 Notebook

以下文件从清理前的 docs/_sources/ 原样复制到 tutorials/upstream/，校验值一致：

| 文件 | SHA-256 |
| --- | --- |
| 1_GO.ipynb | 56cbc99941a6a34e4ad8f82ba806a231ff535c56c964f45988b72178566e1b24 |
| 2_GO_term_embedding.ipynb | 177dac0e448a9c22f038ef6823ad1984ac2a525b96615740ae6ce41442642e52 |
| 3_PhenoMiner_plant.ipynb | b7b1f6dfb2c4c8dedcc4fbc019221674ef0f0e67661e67856573604e435d17fc |

tutorials/local_*.ipynb 是当前环境中的本地教学版本；
tutorials/upstream/ 用于核对上游实现，不作为小鼠分析入口。

## 本次文档清理

清理前本地提交：8a13024。
旧 docs 的 177 个文件全部受 Git 跟踪，清理前没有未提交修改。
删除的是旧网页构建产物、静态图片和前端文件、安装说明副本及报错日志。
已保留三个原始 Notebook。旧文档也可从该提交的 docs/ 路径恢复。

运行依赖位于 models/、resources/、data/go/ 和环境 pathak，不从旧 docs 加载。

## MELA 程序表

- 官方来源：https://mela.wi.mit.edu/download/
- 本次复制来源：CPU:/hy-tmp/GSE342607/author_metadata/annotated_programs.csv
- 作者程序数：70
- SHA-256：1489862271bbeb10a1dc9f0eaf7355d131fdfdfdb8455a55b3e2030dca9523c4

此表是作者已发表程序的参考数据，不能将其重新解释为本项目新发现。
