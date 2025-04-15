from flask import Flask, render_template, request, jsonify
from poem_generator import PoemGenerator
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    logger.info("正在初始化诗歌生成器...")
    generator = PoemGenerator()
    logger.info("诗歌生成器初始化完成")
except Exception as e:
    logger.error(f"初始化诗歌生成器时出错: {str(e)}")
    generator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if generator is None:
            raise Exception("诗歌生成器未正确初始化")

        data = request.get_json()
        start_text = data.get('start_text')
        temperature = data.get('temperature', 0.7)
        
        logger.info(f"收到生成请求: start_text={start_text}, temperature={temperature}")
        
        result = generator.generate_poem(
            start_text=start_text,
            temperature=temperature
        )
        
        logger.info("诗歌生成成功")
        return jsonify({
            'success': True,
            'poem': result['formatted_poem']
        })
    except Exception as e:
        logger.error(f"生成诗歌时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 