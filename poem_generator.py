from transformers import GPT2LMHeadModel, BertTokenizer
import torch
import random
import re

class PoemGenerator:
    def __init__(self, model_path="gpt2-chinese-poem"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # 平仄字典（这里只包含部分常用字，实际应用中需要更完整的字典）
        self.pingze_dict = {
            '平': ['一', '三', '五', '七', '九', '东', '冬', '江', '支', '微', '鱼', '虞', '齐', '佳', '灰', '真', '文', '元', '寒', '删', '先', '萧', '肴', '豪', '歌', '麻', '阳', '庚', '青', '蒸', '尤', '侵', '覃', '盐', '咸'],
            '仄': ['二', '四', '六', '八', '十', '董', '肿', '讲', '纸', '尾', '语', '麌', '荠', '蟹', '贿', '轸', '吻', '阮', '旱', '潸', '铣', '筱', '巧', '皓', '哿', '马', '养', '梗', '迥', '有', '寝', '感', '俭', '豏']
        }
        
    def load_model(self):
        if self.model is None:
            from transformers import GPT2LMHeadModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            
    def get_pingze(self, char):
        """获取单个字的平仄"""
        for category, chars in self.pingze_dict.items():
            if char in chars:
                return category
        return None  # 如果字典中没有这个字，返回None
    
    def check_pingze_pattern(self, line, pattern):
        """检查一行诗是否符合平仄模式"""
        if len(line) != len(pattern):
            return False
        for i, char in enumerate(line):
            pingze = self.get_pingze(char)
            if pingze is None:  # 如果字典中没有这个字，跳过检查
                continue
            if pattern[i] == '平' and pingze != '平':
                return False
            if pattern[i] == '仄' and pingze != '仄':
                return False
        return True
    
    def check_wuyan_pingze(self, lines):
        """检查五言绝句的平仄"""
        if len(lines) != 4:
            return False
            
        # 五言绝句的基本平仄模式（以首句平起为例）
        patterns = [
            '平仄仄平平',  # 第一句
            '仄仄仄平平',  # 第二句
            '仄仄平平仄',  # 第三句
            '平平仄仄平'   # 第四句
        ]
        
        # 检查每句是否符合平仄模式
        for i, line in enumerate(lines):
            if not self.check_pingze_pattern(line, patterns[i]):
                return False
        return True
    
    def generate_random_start(self):
        """生成随机的四字开头"""
        common_starts = [
            "春风十里", "明月几时", "山高水长", "云淡风轻",
            "花开花落", "岁月如歌", "天涯海角", "碧水青山",
            "烟雨江南", "落花流水", "清风明月", "长亭古道"
        ]
        return random.choice(common_starts)
    
    def is_chinese(self, text):
        """检查文本是否只包含中文"""
        return all('\u4e00' <= char <= '\u9fff' for char in text)
    
    def clean_text(self, text):
        """清理文本，只保留中文"""
        # 移除所有非中文字符（包括标点、空格、数字等）
        return re.sub(r'[^\u4e00-\u9fff]', '', text)
    
    def check_rhyme(self, lines):
        """检查诗句是否押韵"""
        if len(lines) < 2:
            return False
        # 获取每句最后一个字
        last_chars = [line[-1] for line in lines]
        # 检查第二句和第四句是否押韵
        return last_chars[1] == last_chars[3]
    
    def format_poem(self, text):
        """格式化古诗，确保符合五言绝句格式"""
        # 清理文本，只保留中文
        text = self.clean_text(text)
        # 按每句5个字分割
        lines = [text[i:i+5] for i in range(0, len(text), 5)]
        # 确保每句都是5个字且只包含中文
        lines = [line for line in lines if len(line) == 5 and self.is_chinese(line)]
        # 只保留前4句（五言绝句）
        lines = lines[:4]
        return lines
    
    def generate_poem(self, start_text=None, max_length=64, temperature=0.7):
        self.load_model()  # 延迟加载模型
        if start_text is None:
            start_text = self.generate_random_start()
            
        for attempt in range(5):
            input_ids = self.tokenizer.encode(start_text, return_tensors='pt')
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,  # 增加避免重复的力度
                    do_sample=True,
                    top_k=40,  # 降低top_k值，使生成更集中
                    top_p=0.9,  # 调整top_p值
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # 增加重复惩罚
                    length_penalty=1.0  # 保持长度惩罚
                )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            lines = self.format_poem(generated_text)
            
            # 检查是否押韵且符合平仄
            if len(lines) == 4 and self.check_rhyme(lines) and self.check_wuyan_pingze(lines):
                return {
                    'start': start_text,
                    'raw_text': generated_text,
                    'formatted_poem': '\n'.join(lines)
                }
        
        # 如果多次尝试后仍未生成押韵的诗，返回最后一次的结果
        return {
            'start': start_text,
            'raw_text': generated_text,
            'formatted_poem': '\n'.join(lines)
        }

def main():
    generator = PoemGenerator()
    
    # 生成5首随机古诗
    print("=== 随机生成古诗 ===")
    for i in range(5):
        result = generator.generate_poem()
        print(f"\n第{i+1}首:")
        print(f"开头: {result['start']}")
        print("生成的诗:")
        print(result['formatted_poem'])
        print("-" * 20)

if __name__ == '__main__':
    main() 