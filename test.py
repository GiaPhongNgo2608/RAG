import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyBoohaB85jrDff9FyXWvbqdcrqHaOEyJXE")

model = genai.GenerativeModel('gemini-pro')

chat = model.start_chat()
response = chat.send_message('Ai là người có thẩm quyền thực hiện rà soát Chiến lược phát triển của PVN/Lĩnh Vực/Đơn vị?')
print(response.text) #  'Here are some suggestions...'
response = model.generate_content("Ai là người có thẩm quyền thực hiện rà soát Chiến lược phát triển của PVN/Lĩnh Vực/Đơn vị?")
print(response.text)
print("done")

with open('/home/rb025/Documents/PVP.txt') as file:
    text = file.read()
sections = text.split('\n')
for i, section in enumerate(sections):
    if i % 2 == 0 and i < (len(sections) - 1):
        response = chat.send_message(section.strip())
        print(f"{section.strip()} \n")
        print(response.text)
