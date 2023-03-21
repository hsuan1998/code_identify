from PIL import Image , ImageDraw , ImageFont
import random
import os

if not os.path.isdir('./captcha_img'):
    os.mkdir('./captcha_img')

#使用字型
font_Reenie = './fonts/ReenieBeanie-Regular.ttf' 
font_Grace = './fonts/CoveredByYourGrace-Regular.ttf'
font_Gloria = './fonts/GloriaHallelujah-Regular.ttf'
fonts_list = [font_Reenie,font_Grace,font_Gloria]

#隨機選擇字型
def random_font():
    fonts = random.choice(fonts_list)
    return fonts

#生成隨機驗證碼(num為碼數)
def getCaptchaText(num):
    captcha_text = ''

    for i in range(num):
        r = chr(random.randint(97,122)) #小寫英文字母
        captcha_text += str(r)
    return captcha_text

for i in range(0,40000):
    #創建空白畫布
    captcha_img = Image.new('RGB',(120,100),'#349cd8')
    #創建畫筆
    drawer = ImageDraw.Draw(captcha_img)
    
    #設定寫入的字型及字體大小
    font_type = random_font()
    size = 0
    x , y = 0 , 0
    if(font_type == './fonts/ReenieBeanie-Regular.ttf'):
        size = 70
        x , y = 10 , 10
    elif(font_type == './fonts/CoveredByYourGrace-Regular.ttf'):
        size = 58
        x , y = 10 , 10
    elif(font_type == './fonts/GloriaHallelujah-Regular.ttf'):
        size = 45
        x , y = 10 , 0
    fonts = ImageFont.truetype(font_type,size)

    #取得寫入的驗證碼
    captcha = getCaptchaText(4)
    #字體顏色
    drawer.ink = 0xffffff
    #寫入圖片
    drawer.text((x,y),captcha,font=fonts)
    #儲存圖片
    captcha_img.save(f'./captcha_img/{captcha}.png')
    
