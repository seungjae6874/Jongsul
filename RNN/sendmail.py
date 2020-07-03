import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

#gmail 아이디, 비번 입력
email_user = 'qkrtmdwo1@gmail.com'
email_password = 'iuvdifbbdgtmnagg'
email_send = 'qkrtmdwo1@gmail.com'

#제목
subject = 'glucose_data'

msg = MIMEMultipart()
msg['From'] = email_user
msg['To'] = email_send
msg['Subject'] = subject

#본문 내용
body = 'Libre를 통한 Glucose data'
msg.attach(MIMEText(body, 'plain'))

#첨부파일 경로/이름 지정
attachment = "종서박_glucose.xlsx"
part = MIMEBase('application','octet-stream')
part.set_payload(open("종서박_glucose.xlsx","rb").read())
encoders.encode_base64(part)
part.add_header('Content-Disposition','attachment', filename=attachment)
msg.attach(part)


text = msg.as_string()
server = smtplib.SMTP('smtp.gmail.com',587)
server.starttls()
server.login(email_user,email_password)

server.sendmail(email_user,email_send,text)

print('email send succed')
server.quit()

