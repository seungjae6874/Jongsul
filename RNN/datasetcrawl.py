from selenium import webdriver
from goodbyecaptcha.solver import Solver
import re, csv
from bs4 import BeautifulSoup
import openpyxl
import time
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from datetime import datetime
from random import uniform, randint
from time import sleep, time
from selenium.webdriver.common.action_chains import ActionChains
import scipy.interpolate as si
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Firefox()

driver.maximize_window()
driver.implicitly_wait(20)

driver.get('https://www.libreview.com')
#
proxy = "127.0.0.1:1000"
auth_details = {"username": "peanut0223@nate.com", "password": "Jongsul123"}
args = ["--timeout 5"]
options = {"ignoreHTTPSErrors": True, "args": args}
client = Solver(
    # With Proxy
    'https://www.libreview.com', sitekey, options=options, proxy=proxy, proxy_auth=auth_details
    # Without Proxy
    # pageurl, sitekey, options=options
)

#



select = Select(driver.find_element_by_id("country-select"))

select.select_by_visible_text('한국')


driver.find_element_by_xpath("//button[@type='submit']").click()


#이제 국적 및 언어 선택 완료 후 로그인 페이지 이동

driver.find_element_by_id('loginForm-email-input').send_keys('peanut0223@nate.com')
driver.implicitly_wait(20)
driver.find_element_by_id('loginForm-password-input').send_keys('Jongsul123')
driver.implicitly_wait(20)

driver.find_element_by_id('loginForm-submit-button').click()



#이제 리브레 장치를 사용하는 유저의 이메일 및 비밀번호로 접속 완료

driver.implicitly_wait(20)
driver.find_element_by_id('main-header-reports-nav-link').click()



driver.implicitly_wait(20)
#이제 유저의 혈당 데이터를 저장해놓은 excel파일을 다운로드

driver.find_element_by_id('exportData-button').click()

#로봇이 아닙니다 체크박스 해결하기

mainwindow = driver.current_window_handle

frame = driver.find_element_by_tag_name("iframe")
driver.switch_to.frame(frame)
driver.find_element_by_id("recaptcha-anchor").click()
driver.switch_to.window(mainwindow)




