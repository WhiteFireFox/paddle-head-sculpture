import paddlehub as hub
import matplotlib.pyplot as plt
import cv2

# 定义头像生成函数
def GetImage(image_adress, style, alpha):
    style_adress = '/home/aistudio/work/' + style + '.jpg'

    stylepro_artistic = hub.Module(name="stylepro_artistic")

    results = stylepro_artistic.style_transfer(
        images=[{
            'content': cv2.imread(image_adress),
            'styles': [cv2.imread(style_adress)]
        }],
        alpha = alpha,
        visualization = True,
    )

    cv2.imwrite('/home/aistudio/transfer.jpg', results[0]['data'])
    
# Style1(简笔黑白)
GetImage(image_adress='image.jpg', style='style1', alpha=0.98)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()

# Style2(简笔彩色)
GetImage(image_adress='image.jpg', style='style2', alpha=1)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()

# Style3(水墨风格)
GetImage(image_adress='image.jpg', style='style3', alpha=0.9)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()

# Style4：Style1(简笔黑白)+Style3(水墨风格)
GetImage(image_adress='image.jpg', style='style1', alpha=0.9)
GetImage(image_adress='transfer.jpg', style='style3', alpha=1)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
