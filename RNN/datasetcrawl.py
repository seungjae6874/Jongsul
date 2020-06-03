from selenium import webdriver
import re, csv
from bs4 import BeautifulSoup
import openpyxl
import time
import numpy as np
import unittest
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
'''
mainwindow = driver.current_window_handle

frame = driver.find_element_by_tag_name("iframe")
driver.switch_to.frame(frame)
driver.find_element_by_id("recaptcha-anchor").click()
driver.switch_to.window(mainwindow)
'''

def write_stat(loops, time):
	with open('stat.csv', 'a', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',',
								quotechar='"', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow([loops, time])  	 
	
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
	
def wait_between(a,b):
	rand=uniform(a, b) 
	sleep(rand)
 
def dimention(driver): 
	d = int(driver.find_element_by_xpath('//div[@id="rc-imageselect-target"]/table').get_attribute("class")[-1]);
	return d if d else 3  # dimention is 3 by default
	
# ***** main procedure to identify and submit picture solution	
def solve_images(driver):	
	WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID ,"rc-imageselect-target"))
        ) 		
	dim = dimention(driver)	
	# ****************** check if there is a clicked tile ******************
	if check_exists_by_xpath('//div[@id="rc-imageselect-target"]/table/tbody/tr/td[@class="rc-imageselect-tileselected"]'):
		rand2 = 0
	else:  
		rand2 = 1 

	# wait before click on tiles 	
	wait_between(0.5, 1.0)		 
	# ****************** click on a tile ****************** 
	tile1 = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH ,   '//div[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]'.format(randint(1, dim), randint(1, dim )))) 
		)   
	tile1.click() 
	if (rand2):
		try:
			driver.find_element_by_xpath('//div[@id="rc-imageselect-target"]/table/tbody/tr[{0}]/td[{1}]'.format(randint(1, dim), randint(1, dim))).click()
		except NoSuchElementException:          		
		    print('\n\r No Such Element Exception for finding 2nd tile')
   
	 
	#****************** click on submit buttion ****************** 
	driver.find_element_by_id("recaptcha-verify-button").click()




mainWin = driver.current_window_handle

frame = driver.find_element_by_tag_name("iframe")
driver.switch_to.frame(frame)
driver.find_element_by_id("recaptcha-anchor").click()
driver.switch_to.window(mainWin)

# ************ switch to the second iframe by tag name ******************
#driver.switch_to_frame(driver.find_elements_by_tag_name("iframe")[1])  
i=1
while i<130:
    print('\n\r{0}-th loop'.format(i))
    driver.switch_to_window(mainWin)
    WebDriverWait(driver,10).until(
        EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME , 'iframe'))
        )
    wait_between(1.0, 2.0)
    if check_exists_by_xpath('//span[@aria-checked="true"]'):
        import winsound
        winsound.Beep(400,1500)
        write_stat(i, round(time()-start)-1)
        break
    driver.switch_to_window(mainWin)
    wait_between(0.3, 1.5)
    driver.switch_to_frame(driver.find_elements_by_tag_name("iframe")[1])
    solve_images(driver)
    i=i+1



