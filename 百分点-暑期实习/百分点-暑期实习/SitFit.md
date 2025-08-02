

🗣 无需提示词或语言器 (verbalisers): 目前的少样本微调技术都需要手工设计的提示或语言器，用于将训练样本转换成适合目标语言模型的格式。SetFit 通过直接从少量标注训练样本中生成丰富的嵌入，完全省去了提示。

🏎 快速训练: SetFit 不需要使用像 T0 或 GPT-3 这样的大规模语言模型就能达到高准确率。因此，典型情况下，它的训练和推理会快一个数量级或以上。

🌎 支持多语言: SetFit 可与 Hub 上的任一 Sentence Tranformer 一起使用，这意味着如果你想让它支持多语言文本分类，你只要简单地微调一个多语言的 checkpoint 就好了。




> SetFit 的两阶段训练过程



SetFit 主要包含两个阶段：首先在少量标注样例 (典型值是每类 8 个或 16 个样例) 上微调一个 Sentence Transformer 模型。然后，用微调得到的 Sentence Tranformer 的模型生成文本的嵌入 (embedding) ，并用这些嵌入训练一个分类头 (classification head) 。

SetFit 利用 Sentence Transformer 的能力去生成基于句对 (paired sentences) 的稠密嵌入。


![]("\img\20230712152646.png")


在第一步微调阶段，它使用对比训练 (contrastive training) 来最大化利用有限的标注数据。首先，通过选择类内 (in-class) 和类外 (out-class) 句子来构造正句对和负句对，然后在这些句对上训练 Sentence Transformer 模型并生成每个样本的稠密向量。

第二步，根据每个样本的嵌入向量和各自的类标签，训练分类头。推理时，未见过的样本通过微调后的 Sentence Transformer 并生成嵌入，生成的嵌入随后被送入分类头并输出类标签的预测。

简言之，第一步使用Sentence Transformer，生成每一个样本（评论）的向量表示，而在第二步则将第一部导出的向量进行分类。

只需要把基础 Sentence Transformer 模型换成多语言版的，SetFit 就可以无缝地在多语言环境下运行。在实验中，SetFit 在德语、日语、中文、法语以及西班牙语中，在单语言和跨语言的条件下，都取得了不错的分类性能。

**注：**对比训练即将正句对x+（分类相同的句子），与负句对x-（分类不同的句子）

比如在政治言论中，0代表不是，1代表是， 0与0，1与1是正键值对。10是负键值对。
![]("\img\20230712152300.png")


> 流程

## 丹麦语

直接硬套的话丹麦语效果不好

	# 导入库
	! pip install datasets
	! pip install sentence_transformers
	! pip install setfit
	! pip install transformers
	from datasets import load_dataset
	from sentence_transformers.losses import CosineSimilarityLoss
	from setfit import SetFitModel, SetFitTrainer


	# 导入数据集
	dataset = load_dataset("danish_political_comments")
	# dataset.save_to_disk("./danish_comments")

查看一下数据集的情况

	dataset

结果，可以看到数据集有三个feature，id sentence 和target，要用的是sentence和target

	DatasetDict({
	    train: Dataset({
	        features: ['id', 'sentence', 'target'],
	        num_rows: 9008
	    })
	})

丹麦数据集不进行数据处理的话一共有四类型标签，1,2,3,4；故随机每类型的标签8份数据。验证集没类型的标签100份数据。

	
	# 分割测试集与验证集
	train_ds = dataset["train"].shuffle(seed=42).select(range(8 * 4))
	test_ds = dataset["train"].shuffle(seed=44).select(range(100 * 4))

	#进行trainer的初始化，注意setFit模型输入参数为text与label，故需要进行转化，否则会报错。
	column_mapping={"sentence": "text", "target": "label"}
	model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
	trainer = SetFitTrainer(
	    model=model,
	    column_mapping=column_mapping,
	    train_dataset=train_ds,
	    eval_dataset=test_ds,
	    loss_class=CosineSimilarityLoss,
	    batch_size=16,
	    num_iterations=20, # Number of text pairs to generate for contrastive learning
	    num_epochs=1 # Number of epochs to use for contrastive learning
	)

训练

	trainer.train()
	metrics = trainer.evaluate()

