# Haggingface

****

## Huggingface是什么 

Hugging Face Hub和 Github 类似，都是Hub(社区)。Hugging Face可以说的上是机器学习界的Github。Hugging Face为用户提供了以下主要功能：

- 模型仓库（Model Repository）：Git仓库可以让你管理代码版本、开源代码。而模型仓库可以让你管理模型版本、开源模型等。使用方式与Github类似。
- 模型（Models）：Hugging Face为不同的机器学习任务提供了许多预训练好的机器学习模型供大家使用，这些模型就存储在模型仓库中。
- 数据集（Dataset）：Hugging Face上有许多公开数据集。


## 基本介绍
hugging face在NLP领域最出名，其提供的模型大多都是基于Transformer的。为了易用性，Hugging Face还为用户提供了以下几个项目：

- Transformers(github, 官方文档): Transformers提供了上千个预训练好的模型可以用于不同的任务，例如文本领域、音频领域和CV领域。该项目是HuggingFace的核心，可以说学习HuggingFace就是在学习该项目如何使用。
- Datasets(github, 官方文档): 一个轻量级的数据集框架，主要有两个功能：①一行代码下载和预处理常用的公开数据集； ② 快速、易用的数据预处理类库。
- Accelerate(github, 官方文档): 帮助Pytorch用户很方便的实现 multi-GPU/TPU/fp16。
- Space：Space提供了许多好玩的深度学习应用，可以尝试玩一下。


## NLP简介

NLP 是语言学和机器学习领域，专注于理解与人类语言相关的一切。
NLP 任务的目标不仅是单独理解单个单词，而且能够理解这些单词的上下文。

常见NLP任务：

- 分类：获取评论的情绪，检测电子邮件是否是垃圾邮件，确定句子语法是否正确或两个句子在逻辑上是否相关
- 对每个单词进行分类：识别句子的语法成分（名词、动词、形容词）或命名实体（人、地点、组织）
- 生成文本内容：用自动生成的文本完成提示，用屏蔽词填充文本中的空白
- 从文本中提取答案：给定问题和上下文，根据上下文中提供的信息提取问题的答案
- 从输入文本生成新句子：将文本翻译成另一种语言，总结文本

NLP的难点：

计算机处理信息的方式与人类不同。 例如，当我们读“我饿了”这句话时，我们可以很容易地理解它的含义。 同样，给定两个句子，例如“我饿了”和“我很难过”，我们可以轻松确定它们的相似程度。 对于机器学习（ML）模型来说，此类任务更加困难。 需要以某种方式处理文本，使模型能够从中学习。

## Huggingface Transformers

- 直接使用预训练模型进行推理
- 提供了大量预训练模型可供使用
- 使用预训练模型进行迁移学习

Transformer 模型用于解决各种 NLP 任务。Transformers 库提供了创建和使用这些共享模型的功能。 模型中心包含数千个预训练模型，任何人都可以下载和使用。 也可以将自己的模型上传到 Hub

Transformers 库中最基本的对象是 pipeline() 函数。 它将模型与其必要的预处理和后处理步骤连接起来，使我们能够直接输入任何文本并获得易于理解的答案

- 示例

		#引入sentiment-analysis 模型
		from transformers import pipeline

		classifier = pipeline("sentiment-analysis")
		classifier("I've been waiting for a HuggingFace course my whole life.")

- 结果

		[{'label': 'POSITIVE', 'score': 0.9598047137260437}]

- 也可以如此使用


		classifier(
		["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"])
		
- 结果

		[{'label': 'POSITIVE', 'score': 0.9598047137260437},
		 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]


该pipeline选定并加载一个预训练模型，可以对语句的情感进行分析。加载后的模型会放入缓存，故再次运行不需要重新加载。


刚刚的pipeline中，具体三个步骤：

- 文本被预处理为模型可以理解的格式。(Tokenzier)
- 预处理后的输入将传递给模型。
- 模型的预测是经过后处理的，因此您可以理解它们。

> 零样本学习（Zero-shot classification）

对尚未进行标记的文本进行分类，这是现实生活中的常见场景，因为注释文本通常非常耗时并需要专业领域知识。对于这个用例，只需要指定标签，因此不必依赖预训练模型的标签。

- 示例

		from transformers import pipeline
		
		classifier = pipeline("zero-shot-classification")
		classifier(
		    "This is a course about the Transformers library",
		    candidate_labels=["education", "politics", "business"],
		)

- 结果
	
	
		{'sequence': 'This is a course about the Transformers library',
		 'labels': ['education', 'business', 'politics'],
		 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}


除此之外，还可用于Text generation等方面，这里不一一赘述了。

> 微调

所有 Transformer 模型（GPT、BERT、BART、T5 等）都已被训练为语言模型。 这意味着他们已经以自我监督的方式接受了大量原始文本的培训。

但是这些模型可能对特定的实际任务并不实用，因此可能需要进行迁移学习，模型会针对给定任务以监督方式（使用人工注释的标签）进行微调。


## Transformers工作原理

预训练是从头开始训练模型的行为：权重随机初始化，并且在没有任何先验知识的情况下开始训练。

![baseToPretrained]("\img\20230711161207.png")


从base model 到Pretrained language model历经极大的消耗，对环境也有极大的污染。

针对已有的预训练模型进行**微调**，即能提高准确率，也能减少训练模型的时间。

例如，可以利用在英语上训练的预训练模型，然后在 arXiv 语料库上对其进行微调，从而产生基于科学/研究的模型。 微调只需要有限数量的数据：预训练模型获得的知识是“转移
的”，因此称为转移学习。

当然，如果拥有巨大量的数据的情况下，从头训练模型也许会更好。


> Transformer 模型结构


![baseToPretrained]("\img\20230711162920.png")

分为两部分：

- Encoder编码器（左）：编码器接收输入并构建其表示（其特征）。 这意味着模型经过优化以从输入中获取理解。
- 解码器（右）：解码器使用编码器的表示（特征）以及其他输入来生成目标序列。 这意味着该模型针对生成输出进行了优化。

按照前面的步骤，Encoder接收来自tokenizer得到的序列化结果，并进一步将其转化为计算机能理解的语言（包含特征等） Decoder接收Encoder的结果并进一步输出为我们能够看懂的结果。

使用场景：

- **Encoder-only models:**适合需要理解输入的任务，例如句子分类和命名实体识别。
- **Decoder-only models:** 适合文本生成等生成任务。
- **Encoder-decoder models or sequence-to-sequence models:** 适合需要输入的生成任务，例如翻译或摘要。

一般来说，Encoder可以独立工作，而Decoder则依赖于Encoder提供的上下文语境等信息。


> **Attention layers**

Transformers的核心之一，让模型关注于特定的文字。

举例来说，翻译英语的"I like this course"，模型需要关注"你"才能获得"喜欢"的正确翻译，因为法文的特性。此外，翻译this时，也需要关注course。

综上，同样的概念适用于与自然语言处理的任何任务，单词的含义受上下文的影响，上下文可以使正在研究的单词之前或者之后的任何单词。




> Trainner

基本参数：

- **model**: Union[PreTrainedModel, nn.Module] = None,

集成了 transformers.PreTrainedMode 或者torch.nn.module的模型

- **args**: TrainingArguments = None,

超参数的定义，这部分也是trainer的重要功能，大部分训练相关的参数都是这里设置的，非常的方便：

- **data_collator**: Optional[DataCollator] = None,

- **train_dataset**: Optional[Dataset] = None,

- **eval_dataset**: Optional[Union[Dataset, Dict[str, Dataset]]] = None,

- **tokenizer**: Optional[PreTrainedTokenizerBase] = None,

- **model_init**: Optional[Callable[[], PreTrainedModel]] = None,

- **compute_metrics**: Optional[Callable[[EvalPrediction], Dict]] = None,

- **callbacks**: Optional[List[TrainerCallback]] = None,

- **optimizers**: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),

- **preprocess_logits_for_metrics**: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,


# 具体流程



![baseToPretrained]("\img\20230714151641.png")

分为三大步：预处理、通过模型传递输入、后处理

> 使用分词器（Tokenzier）进行预处理

- 将输入拆分为单词、子单词或者符号，成为标记（token）
- 将每一个token映射到一个整数
- 添加可能对模型有用的其它输入

所有预处理都需要以模型训练完全相同的方式完后才能完成，需要从Model Hub中下载相关信息。并且使用AutoTokenizer类及其方法from_pretrained()。自动获取与模型标记器相关的数据，并进行缓存。

- 动手尝试
![]("\img\20230714152732.png")

> 使用模型

我们可以像使用标记器一样下载预训练模型。Transformers提供了一个AutoModel类，该类还具有from_pretrained()方法：

	from transformers import AutoModel

	checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
	model = AutoModel.from_pretrained(checkpoint)

这个架构只包含基本转换器模块：给定一些输入，它输出我们将调用的内容隐藏状态（hidden states），亦称特征（features）。对于每个模型输入，我们将检索一个高维向量，表示Transformer模型对该输入的上下文理解。


虽然这些隐藏状态本身可能很有用，但它们通常是模型另一部分（称为头部(head)）的输入。 在Chapter 1中，可以使用相同的体系结构执行不同的任务，但这些任务中的每个任务都有一个与之关联的不同头。


Transformers模块的矢量输出通常较大。它通常有三个维度：

- Batch size: 一次处理的序列数（在我们的示例中为2）。
- Sequence length: 序列的数值表示的长度（在我们的示例中为16）。
- Hidden size: 每个模型输入的向量维度。

输入上图中tokenizer的输出的值并作为Transformers的输入。

![]("/img/20230714153552.png")


> 模型头

模型头将Transformer的输出作为输入，并将其投影到不同的维度，一般如下图所示：

![]("/img/20230714154152.png")

Embedding和Layers组成Transformer模型。嵌入Embedding层将标记化输入的每个输入ID转换为表示关联标记(token)的向量。后续层使用注意机制操纵这些向量，以生成句子的最终表示。


对于我们的示例，我们需要一个带有序列分类头的模型（能够将句子分类为肯定或否定）。因此，我们实际上不会使用AutoModel类，而是使用AutoModelForSequenceClassification，因为AutoModel类只会传入一个模型，将其转化为向量，却没有进行分类操作，无法成为有价值的结果。

	from transformers import AutoModelForSequenceClassification
	
	checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
	model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
	outputs = model(**inputs)

观察输出，可以看到输出了一个2*2的向量，对应两个句子的两个标签。

![]("/img/20230714154852.png")

> 后处理

模型的输出看起来没有什么意义，我们的模型预测第一句为[-1.5607, 1.6123]，第二句为[ 4.1692, -3.3464]。这些不是概率，而是logits，即模型最后一层输出的原始非标准化分数。

想要转换为概率他们需要经过SoftMax层

	import torch
	
	predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
	print(predictions)

这里就可以看出他们预测的概率

第一句为[0.0402, 0.9598]，第二句为[0.9995, 0.0005]。这些是可识别的概率分数。


![]("/img/20230714155255.png")

可使用下列代码查看标签的具体意思：

	model.config.id2label


> 模型

创建与使用模型，使用AutoModel类，其对库中各种可用模型的简单包装。

**初始化BERT模型**

加载配置对象
	from transformers import BertConfig, BertModel
	
	# Building the config
	config = BertConfig()
	
	# Building the model from the config
	model = BertModel(config)

config包含很多用于构建模型的属性，比如说hiddensize属性定义了hidden状态向量的大小，num_hidden_layers定义了Transformer模型的层数。

![]("/img/20230714161802.png")

该模型可以在这种状态下使用，但会输出胡言乱语；首先需要对其进行训练。这将需要很长的时间和大量的数据，并将产生不可忽视的环境影响。为了避免不必要的重复工作，必须能够共享和重用已经训练过的模型。


保存模型和加载模型一样简单—我们使用 save_pretrained() 方法，类似于 from_pretrained() 方法：

	model.save_pretrained("directory_on_my_computer")

将模型与模型属性两个文件保存


## Tokenizer

标记器(Tokenizer)是 NLP 管道的核心组件之一。它们有一个目的：将文本转换为模型可以处理的数据。由于模型只能处理数字，而输入一般为文本，故需要Tokenzier。

### Tokenzier例子

- 1.基于词的(Word-based)

	![]("/img/20230717100204.png")
	以词拆分文本
	
	可以通过应用Python的split()函数，使用空格将文本标记为单词

	使用这种标记器，我们最终可以得到一些非常大的“词汇表”，其中词汇表由我们在语料库中拥有的独立标记的总数定义。

	每个单词都分配了一个 ID，从 0 开始一直到词汇表的大小。该模型使用这些 ID 来识别每个单词。每一个单词都会有一个标识符，这将生成大量的标志，且当词义相近时（比如dog和dogs）模型很难区分。最后，还需要一个自定义标记（token）来表示词汇中单词，这被称为“未知”标记(token)，通常表示为“[UNK]”或”“

- 2.基于字符(Character-based)

	将文本拆分为字符，而不是单词，通常有两个好处：
	
	- 词汇量小得多
	- 词汇外（未知）标记少的多，因为每个单词都可以从字符创建。

	这种方法也并非完美，由于现在表示的是基于字符而不是单词，可能意义不大，每个字符也没有多大意义。这又因语言而定，比如说，中文中，每个字符比拉丁语言中的字符包含更多信息。
	
	另外，我们模型会最终处理大量的token，虽然使用基于单词的tokenizer，单词只是单个标记，转化为字符时，它很容易编程10个或者更多的字符。

- 3.字词标记化

	不将常用词拆分更小子词，将稀有词分解为更有意义的子词。

	比如，annoyingly可能认为是一个罕见的词，分解为 annoying 和ly

	这种方法在土耳其语等粘着型语言(agglutinative languages)中特别有用，可以通过将子词串在一起来形成（几乎）任意长的复杂词。

	

**加载和保存**

	from transformers import BertTokenizer

	tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

使用

	tokenizer("Using a Transformer network is simple")

保存

	tokenizer.save_pretrained("directory_on_my_computer")


全部过程：

- 1.标记化


		from transformers import AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

		sequence = "Using a Transformer network is simple"
		tokens = tokenizer.tokenize(sequence)

		print(tokens)

	此方法输出token

		['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

- 2.由词符（token）到输入ID

	输入ID的转化有tokenzier的convert_tokens_to_ids()方法实现：
		
		ids = tokenizer.convert_tokens_to_ids(tokens)
		print(ids)

	结果：

		[7993, 170, 11303, 1200, 2443, 1110, 3014]

- 3.解码：从词汇索引中，我们想要得到一个字符串，这可以通过decode()实现，如下：

		decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
		print(decoded_string)

	结果：

		'Using a Transformer network is simple'


## 微调一个预训练模型

使用MRPC数据集作为示例，有5801对句子，每个句子有一个标签，指示他们是否同义


导入数据集
![]("/img/20230717164314.png")

简单查看一下
![]("/img/20230717164444.png")


导入tokenizer
![]("/img/20230717173400.png")

在话传给模型前，对两句话需要进行预处理
![]("/img/20230717173522.png")


可以看到，包含了输入词id(input_ids) 和 注意力遮罩(attention_mask) ，讨论类型标记ID(token_type_ids)。在这个例子中，类型标记ID(token_type_ids)的作用就是告诉模型输入的哪一部分是第一句，哪一部分是第二句。

转换回来：
![]("/img/20230717173726.png")

用类型标记ID对BERT进行预训练,并且使用第一章的遮罩语言模型，还有一个额外的应用类型，叫做下一句预测. 这项任务的目标是建立成对句子之间关系的模型。

	tokenized_dataset = tokenizer(
	    raw_datasets["train"]["sentence1"],
	    raw_datasets["train"]["sentence2"],
	    padding=True,
	    truncation=True,
	)

缺点：返回字典（字典的键是输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 类型标记ID(token_type_ids)，字典的值是键所对应值的列表）而且只有当在转换过程中有足够的内存来存储整个数据集时才不会出错


为了将数据保存为数据集，我们将使用Dataset.map()方法，如果我们需要做更多的预处理而不仅仅是标记化，那么这也给了我们一些额外的自定义的方法。这个方法的工作原理是在数据集的每个元素上应用一个函数，因此让我们定义一个标记输入的函数：

	def tokenize_function(example):
	    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

这个函数输入一个字典，返回包括输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 类型标记ID(token_type_ids) 键的新字典。如果键所对应的值包含多个句子，那么依旧可以工作。在map()时可以使用batched=True，显著增加标记速度。

注意：在标记函数中省略了Padding参数，因为填充到最大长度效率不高。

![]("/img/20230717175952.png")

可见向数据集中添加了新的键。



最后一件我们需要做的事情是，当我们一起批处理元素时，将所有示例填充到最长元素的长度——我们称之为动态填充。

### 动态填充

负责在批处理中的数据整理为一个batch的函数成为collate函数，构建DataLoader是传递一个参数，默认是一个函数，将数据转化为PyTorch张量，并将其拼接。

为了解决句子长度统一的问题，必须定义一个collate函数，该函数会每一个batch句子填充到正确的长度。transformers库通过DataCollatorWithPadding为我们提供了这样的参数，实例化时，需要一个标记器（用来知道使用哪个词来填充，以及模型期望填充在左边或者右边）