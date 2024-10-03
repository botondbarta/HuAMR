

sentence_to_amr_prompt = """### Instruction
Provide the AMR graph for the following sentence. Ensure that the graph captures the main concepts, the relationships between them, and any additional information that is important for understanding the meaning of the sentence. Use standard AMR notation, including concepts, roles, and relationships.

### Sentence
{}

### AMR Graph
{}"""

amr_to_sentence_prompt = """### Instruction
Generate a natural language sentence that accurately represents the given AMR graph. Ensure that the sentence captures all the main concepts, relationships, and information present in the AMR notation.

### AMR Graph
{}

### Sentence
{}"""