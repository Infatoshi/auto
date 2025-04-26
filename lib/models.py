import os
import openai

class Models:
    def __init__(self):
        """"
        supported models:
        openai/gpt-4.1
        openai/o3
        openai/o4-mini-high
        x-ai/grok-3-beta
        x-ai/grok-3-mini-beta
        deepseek/deepseek-chat-v3-0324:free
        deepseek/deepseek-r1
        google/gemini-2.5-pro-preview-03-25
        google/gemini-2.5-flash-preview:thinking
        anthropic/claude-3.5-sonnet
        anthropic/claude-3.7-sonnet
        anthropic/claude-3.7-sonnet:thinking
        """
        
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # check if all keys exist and print the ones that do with a unicode checkmark and X for the ones that don't
        if self.openrouter_api_key:
            print("✅ OpenRouter API Key: ", self.openrouter_api_key[:10])
        else:
            print("❌ OpenRouter API Key: ", self.openrouter_api_key)
   
    def _call_openrouter(self, prompt, temperature=0.7, max_tokens=4096, model="openai/gpt-4.1"):
        """Internal method to handle OpenRouter API calls."""
        client = openai.OpenAI(
            api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def gpt_4_1(self, prompt, temperature=0.7, max_tokens=4096, model="openai/gpt-4.1"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def o3(self, prompt, temperature=0.7, max_tokens=4096, model="openai/o3"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def o4_mini_high(self, prompt, temperature=0.7, max_tokens=4096, model="openai/o4-mini-high"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def grok_3_beta(self, prompt, temperature=0.7, max_tokens=4096, model="x-ai/grok-3-beta"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def grok_3_mini_beta(self, prompt, temperature=0.7, max_tokens=4096, model="x-ai/grok-3-mini-beta"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def deepseek_chat_v3(self, prompt, temperature=0.7, max_tokens=4096, model="deepseek/deepseek-chat-v3-0324:free"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def deepseek_r1(self, prompt, temperature=0.7, max_tokens=4096, model="deepseek/deepseek-r1"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def gemini_2_5_pro(self, prompt, temperature=0.7, max_tokens=4096, model="google/gemini-2.5-pro-preview-03-25"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def gemini_2_5_flash(self, prompt, temperature=0.7, max_tokens=4096, model="google/gemini-2.5-flash-preview:thinking"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def claude_3_5_sonnet(self, prompt, temperature=0.7, max_tokens=4096, model="anthropic/claude-3.5-sonnet"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def claude_3_7_sonnet(self, prompt, temperature=0.7, max_tokens=4096, model="anthropic/claude-3.7-sonnet"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)

    def claude_3_7_sonnet_thinking(self, prompt, temperature=0.7, max_tokens=4096, model="anthropic/claude-3.7-sonnet:thinking"):
        return self._call_openrouter(prompt, temperature, max_tokens, model)


