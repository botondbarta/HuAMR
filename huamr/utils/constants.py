sentence_to_amr_prompt = """### Instruction
Provide the AMR graph for the following sentence. Ensure that the graph captures the main concepts, the relationships between them, and any additional information that is important for understanding the meaning of the sentence. Use standard AMR notation, including concepts, roles, and relationships.

### Sentence
{}

### AMR Graph
{}"""

SHORTER_PROMPT = "Text: {}\nAMR: {}"

SYSTEM_PROMPT = "You are an AMR parser that converts natural language sentences into their Abstract Meaning Representation (AMR) graphs. Given a sentence, generate a well-structured AMR graph that accurately represents its meaning. Capture key concepts, semantic roles, and relationships using standard AMR notation. Ensure that the output is syntactically correct, includes core arguments (e.g., ARG roles), modifiers (e.g., time, location, manner), named entities, and any necessary re-entrancies to reflect co-references. Maintain consistency with AMR conventions and strive for linguistic accuracy."