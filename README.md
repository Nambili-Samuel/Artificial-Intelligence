# What is Artificial Intelligence (AI)?

![Image Description](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg4Cd9OPlBfhoiohbzhYtbOdlC3kuMQVNTPbyYsL2ocItAO6u6ataBv2NfHuLGzkOtmrGYSPkxoiEjfBlNKw0OPdI37ldFBJLEi5UDQyvu4blWxqrMmKF5Dqip79mthRKSnNIkVQbDfjCScd08ZDkKG_PqwGKi3-ncEi1D1EC_G-mycqL0i3ntP2_1i4rzt/s16000/AI.webp)

[Artificial intelligence (AI)](https://cloud.google.com/learn/what-is-artificial-intelligence?hl=en) is a technology that enables computers and machines to simulate human capabilities such as learning, comprehension, problem-solving, decision-making, creativity, and autonomy. AI-powered applications and devices like [Amazon Alexa](https://alexa.amazon.com/), Apple’s [Siri](https://www.apple.com/siri/), [Google Assistant](https://assistant.google.com/), and Microsoft’s [Cortana](https://support.microsoft.com/en-us/topic/end-of-support-for-cortana-d025b39f-ee5b-4836-a954-0ab646ee1efa) can interact with users, recognize objects, understand and respond to human language, and adapt based on new information to improve user experience.

AI systems can operate independently, reducing the need for human intervention—a classic example being a self-driving car. However, in 2024, much of the AI community and AI-related innovations focus on breakthroughs in [generative AI](https://www.nvidia.com/en-us/glossary/generative-ai/) (gen AI), which can create original content like text, images, and videos. To understand generative AI, it’s helpful first to understand the foundational technologies of [machine learning](https://www.ibm.com/topics/machine-learning) (ML) and [deep learning](https://www.deeplearning.ai/), which underpin these tools. AI can also be categorized based on the methods used for training models. Below are the common types of learning techniques in AI:

### 1. Supervised Learning:
- **Definition**: In [Supervised learning](https://cloud.google.com/discover/what-is-supervised-learning?hl=en), the AI model is trained using labeled data, where the input data is paired with the correct output. The model learns by comparing its predictions to the actual outcomes and adjusts based on errors.
- **Examples**: Spam email classification, image recognition (e.g., identifying cats vs. dogs), and regression tasks like predicting house prices.
- **Use Case**: It's ideal for problems where the desired output is known and available, and the goal is to learn a mapping from inputs to outputs.

### 2. Unsupervised Learning:
- **Definition**: Unsupervised learning involves training a model on data that has no labels. The AI attempts to find hidden patterns or structures in the data without any explicit guidance on what the outputs should be.
- **Examples**: Clustering (e.g., customer segmentation), dimensionality reduction (e.g., PCA), and anomaly detection (e.g., fraud detection).
- **Use Case**: It’s useful when you have large amounts of unlabelled data and want to find underlying structures or groupings.

### 3. Semi-Supervised Learning:
- **Definition**: [Semi-supervised learning](https://www.ibm.com/topics/semi-supervised-learning) is a hybrid approach where the model is trained on a combination of a small amount of labeled data and a large amount of unlabeled data. It leverages the strengths of both supervised and unsupervised learning.
- **Examples**: Labeling web pages, facial recognition (using few labeled examples), and medical image classification where labeled data is scarce but unlabeled data is abundant.
- **Use Case**: This is helpful when acquiring labeled data is expensive or time-consuming.

### 4. Reinforcement Learning:
- **Definition**: In reinforcement learning, an agent learns to make decisions by interacting with an environment. It receives feedback in the form of rewards or penalties based on the actions it takes and aims to maximize its cumulative reward over time.
- **Examples**: Game-playing AI (e.g., AlphaGo), robotic control, self-driving cars, and recommendation systems.
- **Use Case**: It’s suitable for tasks where an agent needs to make a series of decisions to achieve long-term goals in dynamic environments.

### 5. Self-Supervised Learning:
- **Definition**: [Self-supervised learning](https://www.ibm.com/topics/self-supervised-learning) is a form of unsupervised learning where the system generates its own labels from the input data. This approach can be thought of as a way for the model to learn from the data without needing external labeling.
- **Examples**: Language models (e.g., GPT, BERT) that learn to predict the next word in a sentence based on context, or predicting missing parts of images.
- **Use Case**: It’s gaining popularity for tasks like natural language processing and computer vision where labeled data is limited.

These learning techniques form the basis of how AI systems are trained and can be combined or adapted to suit different types of problems.

---

## Machine learning

Within AI, we have **machine learning**, a subfield of artificial intelligence that gives computers the ability to learn without explicitly being programmed for specific tasks. There are many types of machine learning techniques or algorithms, including linear regression, logistic regression, decision trees, random forests, support vector machines (SVMs), [k-nearest neighbors (KNN)](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn), clustering, and more. Each of these approaches is suited to different kinds of problems and data.

The goal of AI is to create computer models that exhibit intelligent behaviors like humans, according to [Boris Katz](https://www.csail.mit.edu/person/boris-katz), an AI researcher and leader of the InfoLab Group at CSAIL, machines can recognize a visual scene, and understand the text written or natural language in the physical world.

One of the most widely used types of machine learning algorithms is the neural network, or [artificial neural network (ANN)](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/artificial-neural-network). Inspired by the structure and function of the human brain, ANNs consist of interconnected layers of nodes, or "neurons." These networks typically include three types of layers:

- The **input layer**, which receives raw data.
- One or more **hidden layers**, where the network processes and transforms information.
- The **output layer**, which produces the final prediction or classification.

Each layer in the network is densely connected to the next, allowing the model to learn complex relationships within data by adjusting the connections between nodes through training. Hidden layers are critical, as they enable the network to capture intricate, non-linear patterns essential for tasks like image and speech recognition, language translation, and other data-intensive applications. This layered architecture, combined with vast data and computational power, makes neural networks highly effective for pattern recognition and predictive analytics.

![Image Description](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiOdccxJxUaTRv7C92tt5x-yC88FEW23K_gIsJ2Z9C_2cmjymWwIE2gNcocW4RFEPYZ6So2q5R2kl-JYKPx-MnFUv-sj7y8LGSEvgqkbNgQXVlK_S8VVewS7l8vcZoX7L6I2_RvommpVCt9MN-KojUg-q-tchzVJseTPkejcfpoDzhgq_ulPiRzsdehaRlm/s16000/rpqrCoZ.png)

The simplest form of machine learning is called supervised learning, which involves the use of labeled data sets to train algorithms to classify data or predict outcomes accurately. In supervised learning, humans pair each training example with an output label. The goal is for the model to learn the mapping between inputs and outputs in the training data, so it can predict the labels of new, unseen data.


### Deep learning

**Deep learning** is a subset of machine learning that utilizes multilayered neural networks, known as **deep neural networks**, which more closely simulate the complex decision-making capabilities of the human brain. Deep neural networks comprise an **input layer**, multiple **hidden layers** (typically three or more, but often hundreds), and an **output layer**. This architecture differs from neural networks in classic machine learning, which generally include only one or two hidden layers.

These multiple layers enable [unsupervised learning](https://www.ibm.com/topics/unsupervised-learning): they can autonomously extract features from large, unlabeled, and unstructured datasets and make independent predictions about the data's content. Since deep learning minimizes the need for human intervention, it supports machine learning on a massive scale. Deep learning is particularly effective for tasks like [natural language processing (NLP)](https://www.ibm.com/topics/natural-language-processing) and [computer vision](https://www.ibm.com/topics/computer-vision), where fast and accurate pattern recognition in vast datasets is essential. Today, some form of deep learning drives most of the AI applications we encounter.

---

## Generative AI

**Generative AI**, sometimes called **gen AI**, refers to deep learning models capable of creating complex, original content—such as long-form text, high-quality images, realistic video or audio, and more—in response to user prompts. At a high level, generative models encode a simplified representation of their training data, drawing from this representation to create new content that’s similar, but not identical, to the original data. Generative models have long been used in statistics to analyze numerical data. However, over the last decade, they have evolved to analyze and generate complex data types, driven by the development of three advanced deep learning model types:

- **[Variational Autoencoders (VAEs)](https://www.ibm.com/think/topics/variational-autoencoder)**, introduced in 2013, which enable models to generate multiple variations of content based on prompts or instructions.

- **Diffusion Models**, first introduced in 2014, which add "noise" to images until they become unrecognizable, then remove the noise to generate original images based on user prompts.

- **[Transformers](https://www.ibm.com/topics/transformer-model?mhsrc=ibmsearch_a&mhq=what%20is%20a%20transformer%20model%26quest%3B)**, trained on sequenced data to generate extended sequences (such as words in sentences, shapes in images, frames in video, or commands in code). Transformers power many of today’s most prominent generative AI tools, including ChatGPT and GPT-4, Copilot, BERT, Bard, and Midjourney.


---

## How generative AI works

In general, **generative AI** operates in three phases:

1. **Training** – Building a foundation model by training on large datasets.
2. **Tuning** – Adapting the model for specific applications.
3. **Generation, Evaluation, and Further Tuning** – Generating outputs, evaluating results, and iteratively tuning to improve accuracy.

---

## Training

Generative AI begins with a **foundation model**—a deep learning model that serves as a base for various generative AI applications.

The most common foundation models today are **[large language models (LLMs)](https://www.ibm.com/topics/large-language-models)**, designed for text generation. However, foundation models also exist for image, video, sound or music generation, and even multimodal models that support multiple types of content.

To create a foundation model, practitioners train a deep learning algorithm on vast volumes of raw, unstructured, and unlabeled data, such as terabytes or petabytes of text, images, or video from the internet. This training yields a **[neural network](https://www.ibm.com/topics/neural-networks)** with billions of **parameters**—encoded representations of entities, patterns, and relationships within the data—enabling it to generate content autonomously in response to prompts.

This training process is highly compute-intensive, time-consuming, and costly, requiring thousands of clustered graphics processing units (GPUs) and several weeks to complete, often costing millions of dollars. **Open source foundation model projects**, such as Meta’s Llama-2, allow generative AI developers to bypass this costly step.


---

## Tunning

Next, the model must be tuned to a specific content generation task. This can be done in various ways, including: Fine-tuning, which involves feeding the model application-specific labeled data—questions or prompts the application is likely to receive, and corresponding correct answers in the wanted format. Reinforcement learning with human feedback (RLHF), in which human users evaluate the accuracy or relevance of model outputs so that the model can improve itself. This can be as simple as having people type or talk back corrections to a chatbot or virtual assistant.

Developers and users regularly assess the outputs of their generative AI apps, and further tune the model—even as often as once a week—for greater accuracy or relevance. In contrast, the foundation model itself is updated much less frequently, perhaps every year or 18 months.

Another option for improving a gen AI app's performance is retrieval augmented generation (RAG), a technique for extending the foundation model to use relevant sources outside of the training data to refine the parameters for greater accuracy or relevance.

## Benefits of AI

AI offers numerous benefits across various industries and applications. Some of the most commonly cited advantages include:

- **Automation of repetitive tasks**
- **More and faster insights from data**
- **Enhanced decision-making**
- **Fewer human errors**
- **24x7 availability**
- **Reduced physical risks**

### Automation of Repetitive Tasks

AI can automate routine, repetitive, and often tedious tasks—ranging from digital tasks such as data collection and entry, to physical tasks like warehouse stock-picking and manufacturing processes. This automation frees employees to focus on higher-value, more creative work.

### Enhanced Decision-Making

Whether used for decision support or fully automated decision-making, AI enables faster, more accurate predictions and reliable, **[data-driven decisions](https://www.ibm.com/think/topics/data-driven-decision-making)**. With automation, AI helps businesses act on opportunities and respond to crises in real time, without human intervention.

### Fewer Human Errors

AI can reduce human errors by guiding people through the proper steps of a process, flagging potential errors, and fully automating processes. This benefit is crucial in fields like healthcare, where AI-guided surgical robotics can provide consistent precision. Machine learning algorithms can also improve accuracy over time by "learning" from more data.

### Round-the-Clock Availability and Consistency

AI operates around the clock, ensuring consistent performance every time. For example, AI chatbots and virtual assistants can reduce staffing demands in customer service, while in manufacturing, AI can maintain consistent quality and output for repetitive tasks.

### Reduced Physical Risk

By automating dangerous work—such as handling hazardous materials, performing tasks underwater, at high altitudes, or even in space—AI reduces the need to expose humans to physical risks. Though still advancing, self-driving vehicles also hold potential to reduce injury risks for passengers.

---

## Applications and Use Cases for Artificial Intelligence

- **Speech Recognition:** Converts spoken language to text.
- **Image Recognition:** Identifies and categorizes elements in an image.
- **Translation:** Translates between languages.
- **Customer Support:** AI-powered chatbots and virtual assistants.
- **Predictive Modeling:** Forecasts specific outcomes based on data.
- **Data Analytics:** Discovers patterns for business intelligence.
- **Cybersecurity:** Detects and responds to cyber threats autonomously.

Some of the real-world applications of AI across various industries:

Companies use AI-powered chatbots and virtual assistants to handle customer inquiries and support tickets, relying on [natural language processing (NLP)](https://www.ibm.com/topics/natural-language-processing) to understand and respond to questions about order status, product details, and return policies. These tools provide around-the-clock support, faster responses to frequently asked questions, and free up human agents to focus on higher-level tasks, improving both service speed and consistency.

Machine learning and deep learning algorithms help identify fraudulent transactions by analyzing transaction patterns for anomalies, such as unusual spending habits or unexpected login locations. This allows organizations to quickly respond to potential fraud, limiting its impact and enhancing customer trust. Retailers, banks, and other customer-facing businesses use AI to deliver personalized marketing experiences. Deep learning algorithms analyze customer purchase histories and behaviors to recommend relevant products and generate tailored marketing content and special offers in real-time, which helps improve customer satisfaction, drive sales, and reduce churn.

AI-driven recruitment platforms streamline hiring by automating resume screening, matching candidates to job descriptions, and even conducting preliminary interviews using video analysis. These tools significantly reduce administrative workload, shorten response times, and improve the overall candidate experience. AI models analyze data from sensors, IoT devices, and operational technology (OT) to predict maintenance needs and anticipate equipment failures. This AI-powered preventive maintenance helps reduce downtime and ensures the business stays ahead of potential supply chain disruptions that could impact operations.

1. [What is Artificial Intelligence (AI)? - Google Cloud] https://cloud.google.com/learn/what-is-artificial-intelligence?hl=en
2. [Introduction to Machine Learning - IBM](https://www.ibm.com/topics/machine-learning)
3. [Deep Learning Specialization - Coursera (by Andrew Ng)](https://www.coursera.org/specializations/deep-learning)
4. [Artificial Neural Network (ANN) Overview - ScienceDirect](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/artificial-neural-network)
5. [Natural Language Processing (NLP) - Stanford University](https://web.stanford.edu/~jurafsky/slp3/)
6. [Generative AI Explained - NVIDIA](https://www.nvidia.com/en-us/glossary/generative-ai/)
7. [AI and Machine Learning - Harvard University](https://online-learning.harvard.edu/subject/artificial-intelligence)
8. [Large Language Models (LLMs) - MIT Technology Review](https://www.technologyreview.com/topic/artificial-intelligence/)
9. [Neural Networks: Concepts and Applications - UC Berkeley](https://inst.eecs.berkeley.edu/~cs182/sp21/)
10. [Data-Driven Decision Making - MIT Sloan Management Review](https://sloanreview.mit.edu/)
11. [Variational Autoencoders - DeepMind Blog](https://deepmind.com/blog/article/understanding-variational-autoencoders)
12. [Introduction to Transformer Models - Stanford CS224N](http://web.stanford.edu/class/cs224n/)
13. [Semi-Supervised Learning - Carnegie Mellon University](https://www.cs.cmu.edu/~semisup/)
14. [Self-Supervised Learning - Google AI Blog](https://ai.googleblog.com/)
15. [Computer Vision: Foundations and Trends - Microsoft AI](https://www.microsoft.com/en-us/ai/computer-vision)
16. [Ethics in AI - Oxford Internet Institute](https://www.oii.ox.ac.uk/research/ai-ethics-and-governance/)
17. [The Future of AI - Brookings Institute](https://www.brookings.edu/topic/artificial-intelligence/)
18. [AI in Healthcare - Mayo Clinic](https://www.mayoclinic.org/tests-procedures/ai-in-healthcare/about/)
19. [AI in Finance - Forbes](https://www.forbes.com/sites/forbestechcouncil/2023/01/01/how-ai-is-transforming-financial-services/)
20. [AI in Retail - McKinsey & Company](https://www.mckinsey.com/industries/retail/our-insights/retail-speaks-seven-imperatives-for-the-industry)
21. [Autonomous Vehicles and AI - Tesla AI](https://www.tesla.com/autopilot)
22. [AI-Powered Chatbots - ChatGPT by OpenAI](https://openai.com/chatgpt)
23. [Predictive Maintenance - General Electric (GE) Digital](https://www.ge.com/digital/applications/predictive-maintenance)
24. [Automation in AI - World Economic Forum](https://www.weforum.org/agenda/ai-and-automation)
25. [Customer Support Automation - Salesforce](https://www.salesforce.com/products/service-cloud/features/ai-customer-support/)
26. [Cybersecurity and AI - Palo Alto Networks](https://www.paloaltonetworks.com/cyberpedia/what-is-artificial-intelligence-in-cybersecurity)
27. [Computer Vision in Retail - MIT Media Lab](https://www.media.mit.edu/projects/computer-vision-in-retail/)
28. [Personalized Marketing with AI - HubSpot](https://blog.hubspot.com/marketing/ai-marketing)
29. [AI in Human Resources - SHRM](https://www.shrm.org/resourcesandtools/hr-topics/technology/pages/artificial-intelligence-hr.aspx)
30. [Generative AI Basics - OpenAI](https://openai.com/research/generative-models)
31. [Diffusion Models in AI - Stanford AI Lab](https://ai.stanford.edu/blog/diffusion-models/)
32. [AI for Fraud Detection - Mastercard](https://www.mastercard.com/news/artificial-intelligence-fraud-detection/)
33. [AI Ethics Guidelines - UNESCO](https://unesdoc.unesco.org/ark:/48223/pf0000373434)
34. [Deep Learning Model Training - NVIDIA Developer Blog](https://developer.nvidia.com/blog/deep-learning/)
35. [AI Model Tuning - Google AI](https://ai.google/education/tuning/)
36. [Foundation Models in AI - Stanford HAI](https://hai.stanford.edu/research/foundation-models)
37. [Llama-2 - Meta AI](https://ai.facebook.com/blog/llama-2/)
38. [Ethics in AI and Society - AI Now Institute](https://ainowinstitute.org/)
39. [AI and Human-Machine Interaction - Georgia Tech](https://research.gatech.edu/human-centered-ai)
40. [Introduction to Artificial Intelligence - MIT OpenCourseWare](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)



--- 

