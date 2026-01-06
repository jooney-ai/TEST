from google import genai
import json
import os

# 1. API 키 설정 (본인의 키로 교체하세요)
# NAME = os.environ.get("NAME", 0)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PROMPT = os.environ.get("PROMPT")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
# genai.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. 모델 설정 (Gemini 1.5 Flash가 빠르고 저렴하여 추천)
# model = genai.GenerativeModel('gemini-2.5-flash')
model = 'gemini-2.5-flash'


def message(text):

    try:
        discord_payload = {"content": text}
        response = requests.post(DISCORD_WEBHOOK_URL, json=discord_payload)
        response.raise_for_status()
        print(f"전송 성공:")
        print(text)
        return True
    
    except Exception as e:
        print(f"전송 실패: {e}")
        return False


def save_memory(video_info):
    try:
        os.makedirs("Memory", exist_ok=True)
        memory = "Memory/test.jsonl"
        with open(memory, 'a', encoding='utf-8') as f:
            f.write(json.dumps(video_info, ensure_ascii=False) + "\n")
        print(f"[save memory]")
    
    except Exception as e:
        print(f"데이터 저장 실패: {e}")
      

def test():
    try:
        # API 호출
        response = client.models.generate_content(
            model=model, 
            contents=prompt
        )
        
        # 텍스트를 JSON으로 변환 (가끔 ```json ``` 태그가 붙을 수 있어 제거)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_text)

        # 결과 처리
        if result.get("alert"):
            text = f"🚨 [긴급 알림] {result['title']} \n 내용: {result['reason']} \n 출처: {result['source']}"
            result = message(text)
            if result:
                save_memory(text)

        else:
            print("✅")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    test()
