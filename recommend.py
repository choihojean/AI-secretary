import gradio as gr
from PIL import Image
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import urllib.request
from urllib.parse import urlencode

load_dotenv()
client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")
googlemaps_api_key = os.getenv("GOOGLEMAPS_API_KEY")
genai.configure()

# DALL-E API를 호출하여 이미지를 생성하는 함수
def get_image_from_dalle(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt, size="1024x1024", quality = "standard", n=1
    )
    
    image_url = response.data[0].url
    urllib.request.urlretrieve(image_url, "generated_image.jpg")
    return "generated_image.jpg"

def process(image, prompt):
    try:
        prompt = """해당 여행지의 정보, 명소, 문화적 특징에 대해서
        한국어로 설명해줘"""

        vmodel = genai.GenerativeModel('gemini-pro-vision')
        response = vmodel.generate_content([prompt, image])
        location_info = response.text

        location = location_info.strip()
        
        # Google Maps URL 생성
        params = {
            'key': googlemaps_api_key,
            'q': location
        }
        googlemaps_url = f"https://www.google.com/maps/embed/v1/place?{urlencode(params)}"

        # DALL-E로 이미지 생성
        dalle_prompt = f"Create an image of a famous landmark or iconic place in {location}"
        landmark_image = get_image_from_dalle(dalle_prompt)

        return location_info, f'<iframe width="450" height="450" style="border:0" loading="lazy" allowfullscreen src="{googlemaps_url}"></iframe>', landmark_image
    except Exception as e:
        return f"에러 발생: {e}", "", ""

# Gradio 인터페이스 설정
demo = gr.Interface(
    fn=process,
    inputs=[gr.Image(type="pil"), "text"],
    outputs=[gr.Textbox(label="여행지 정보"), 
             gr.HTML(label="지도"),
             gr.Image(label="여행지 랜드마크")],
    title="여행지 정보를 물어보세요!",
    description="사진을 첨부하시면 사진 속 여행지의 위치와 명소를 알려드려요!"
)

# Gradio 인터페이스 실행
if __name__ == "__main__":
    demo.launch()
