import gradio as gr
from PIL import Image
import urllib.request
from urllib.parse import urlencode
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from io import BytesIO
import os
from googleapiclient.discovery import build

# .env
# OPENAI_API_KEY=YOUR_OPENAI_API_KEY
# GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
# GOOGLEMAPS_API_KEY=YOUR_GOOGLEMAPS_API_KEY
# YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY
load_dotenv()

client = OpenAI()
genai.configure()
googlemaps_api_key = os.getenv("GOOGLEMAPS_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

input_image_width = 150  # 입력받는 이미지 칸 크기
output_image_size = 512  # 출력하는 이미지 크기

system_prompt = """
    먼저 이 위치의 지명을 말해줘.
    예시) '여기는 제주도입니다.'
    
    다음으로 이 위치에 대한 정보, 명소, 문화적 특징에 대해서 설명해줘.
"""

# gemini-pro-vision, gpt-3.5-turbo를 이용한 답변 생성
def Process(prompt, history, image) -> str:
    messages = [{"role": "system", "content": system_prompt if len(history) == 0 else ""}]
    
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    messages.append({"role": "user", "content": prompt})
    
    if image is not None and len(history) == 0:
        vmodel = genai.GenerativeModel('gemini-pro-vision')
        response = vmodel.generate_content(["위치가 어디야? 간단하게 알려줘.", image])

        location = response.parts[0].text.strip()  # OS의 문제로 추정 // windows OS는 위의 주석 처리한 코드 사용
        messages.insert(1, {"role": "system", "content": location})
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return completion.choices[0].message.content

#생성된 컨텐츠 저장을 위한 변수
map_html = ""
generated_image = None
video_html = ""

# Google Map API를 이용한 지도 출력
def Map(chatbot) -> str:
    global map_html
    # 첫 번째 질문에 대한 응답에만 작동
    if len(chatbot) == 1:
        # chatbot = [[res, req], ...]
        text = chatbot[-1][1]
        
        # text 값이 없을 때의 예외 처리
        if text is None:
            return ''
        
        location = ''.join(map(str, list(text.split('.', 1)[0])[4:-3]))
        
        # Google Maps URL 생성
        params = {
            'key': googlemaps_api_key,
            'q': location
        }
        googlemaps_url = f"https://www.google.com/maps/embed/v1/place?{urlencode(params)}"
        
        map_html = f'<iframe width="512" height="450" style="border:0" loading="lazy" allowfullscreen src="{googlemaps_url}"></iframe>'
    return map_html

# dall-e-3를 이용한 이미지 생성
def GetImage(chatbot):
    global generated_image
    # 첫 번째 질문에 대한 응답에만 작동
    if len(chatbot) == 1:
        # chatbot = [[res, req], ...]
        text = chatbot[-1][1]
        
        # text 값이 없을 때의 예외 처리
        if text is None:
            return "./mapfolding.webp"
        
        location = ''.join(map(str, list(text.split('.', 1)[0])[4:-3]))
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Create an image of a famous landmark or iconic place in {location}",
            size="1024x1024",
            quality="standard",
            n=1
        )

        image_url = response.data[0].url
        with urllib.request.urlopen(image_url) as url:
            image_data = url.read()
        generated_image = Image.open(BytesIO(image_data))
    return generated_image

# YouTube API를 이용한 비디오 검색 및 출력
def GetVideo(chatbot):
    global video_html
    # 첫 번째 질문에 대한 응답에만 작동
    if len(chatbot) == 1:
        # chatbot = [[res, req], ...]
        text = chatbot[-1][1]
        
        # text 값이 없을 때의 예외 처리
        if text is None:
            return ''
        
        location = ''.join(map(str, list(text.split('.', 1)[0])[4:-3]))
        
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # 검색 쿼리 설정
        request = youtube.search().list(
            part="snippet",
            q=f"{location} travel",
            type="video",
            relevanceLanguage="ko",  # This line ensures the results are relevant to Korean language
            maxResults=1
        )
        response = request.execute()
        
        # 비디오 ID 추출
        video_id = response['items'][0]['id']['videoId']
        
        video_html = f'<iframe width="512" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
    return video_html

# gradio UI 구성
with gr.Blocks(title="여행 챗봇") as demo:
    with gr.Row():
        # 좌측 UI 구성
        with gr.Column():
            # ChatInterface
            chat = gr.ChatInterface(
                fn=Process,
                # Image
                additional_inputs=gr.Image(height=input_image_width, sources='upload', type="pil", label="이미지"),
                retry_btn=None,
                undo_btn=None,
                clear_btn=None)
            chat.chatbot.height = 400
            chat.chatbot.label = "여행 챗봇"

            # 새로고침 버튼 추가
            refresh_button = gr.Button("새 여행지 질문하기")
            refresh_button.click(fn=None, js="window.location.reload()")
            
        # 우측 UI 구성
        with gr.Column():
            # 지도
            html = gr.HTML(label="지도")
            chat.chatbot.change(fn=Map, inputs=chat.chatbot, outputs=html)
            
            # 이미지
            image = gr.Image(value=None, height=output_image_size, width=output_image_size, label="여행지 랜드마크")
            chat.chatbot.change(fn=GetImage, inputs=chat.chatbot, outputs=image)
            
            # 비디오
            video = gr.HTML(label="여행지 소개 영상")
            chat.chatbot.change(fn=GetVideo, inputs=chat.chatbot, outputs=video)

# app 실행
if __name__ == "__main__":
    demo.launch()
