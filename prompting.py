class PromptingEngine:
    def __init__(self,strategy: str = "zero-shot", examples: list = None):
        self.system_prompt = None
        self.strategy = strategy
        self.conversation = []
        self.examples = examples or []

    def set_strategy(self, strategy: str):
        """Update the prompting strategy (zero-shot, few-shot, etc.)"""
        self.strategy = strategy

    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt for the conversation"""
        self.system_prompt = system_prompt
        self.conversation.append({"role": "system", "content": system_prompt})

    def add_example(self, input_text: str, output_text: str):
        """Add example for few-shot prompting"""
        self.examples.append((input_text, output_text))

    def generate_prompt(self,dialect, input_text: str) -> str:
        """Construct the full prompt based on strategy"""
        if self.strategy == "zero-shot":

            input_text = self.get_dialect_specific_prompt(dialect, input_text)
            self.conversation.append({"role": "user", "content": input_text})

        elif self.strategy == "few-shot-conversational":
            for example in self.examples:
                input_text = self.get_dialect_specific_prompt(dialect, input_text)
                self.conversation.append({'role':'assistant', 'content':example})
                self.conversation.append({"role": "user", "content": input_text})

        elif self.strategy == "few-shot":
            input_text = self.get_dialect_specific_prompt(dialect, input_text)
            input_text = f"{input_text}\n \n ### Here are some examples of the {dialect} dialect:\n"
            for example in self.examples:
                input_text += f"""### {example} \n """
            self.conversation.append({"role": "user", "content": input_text})
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    

    def get_dialect_specific_prompt(self, dialect: str, prompt) -> str:
        """Get the dialect-specific prompt"""
        if dialect == "Basel":
            prompt = prompt.replace("#dialect_insert#", "Basel dialect (Baseldiitsch)")
        elif dialect == "Zürich":
            prompt = prompt.replace("#dialect_insert#", "Zürich dialect (Züridüütsch)")
        elif dialect == "Bern":
            prompt = prompt.replace("#dialect_insert#", "Bern dialect (Bärndütsch)")
        elif dialect == "Luzern":
            prompt = prompt.replace("#dialect_insert#", "Luzern dialect (Lozärnerdütsch)")
        elif dialect == "St. Gallen":
             prompt = prompt.replace("#dialect_insert#", "St. Gallen dialect (Sanggallerdütsch)")
        else:
            raise ValueError(f"Unknown dialect: {dialect}")
        
        return prompt

        
        

