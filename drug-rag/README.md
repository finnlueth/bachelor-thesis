# Drug Repurposing RAG

We have a protein (or even specific domain) we wish to target.\
Is it possible to find an existing small molecule to do the job?

Risk: High\
Difficulty: High

## Idea

* Find or train model that captures "semantic" information between ligands and proteins
  * RAG + LLM?
  * RAG + Diffusion?
  * RAG + Flow Matching?
* Instead of retrieving text/nlp information, how do we teach RAG to retrieve compounds
* First Step: Adapt RAG system to idenify possible drugs to target known protein
* Reverse Step: Find proteins targeted by one drug
* Mechanism for model to say "IDK WTF is this"
* Retrive "top-k" candidates

## Various aspects

* Create structural alphabet appendix to 3Di to encode drug structures?
* Select know molecules from libraries
  * Approved compounds (Repurposing)
  * (Easily) synthesizable compounds
  * Other Known compounds
  * Compound library (in Lead Discovery)
  * Attach live databases i.e. "web search" across proprietary DBs

## Links

* https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold3
* https://www.moml.mit.edu/submit
* https://polarishub.io/

### RAG

* https://github.com/lnairGT/Diffusion-with-RAG
* https://aws.amazon.com/blogs/machine-learning/improve-your-stable-diffusion-prompts-with-retrieval-augmented-generation/
* https://www.llamaindex.ai/
* https://cohere.com/blog/command-r
* https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
* https://github.com/microsoft/graphrag
* https://docs.ragas.io/en/stable/
* https://www.youtube.com/watch?v=qN_2fnOPY-M
* https://www.youtube.com/watch?v=sVcwVQRHIc8

### Drug Discovery

* https://en.wikipedia.org/wiki/Hit_to_lead
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3058157/