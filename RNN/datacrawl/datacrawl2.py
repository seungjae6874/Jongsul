import time # delay
import urllib.parse # urlcomponent
import pyperclip # clipboard
from selenium import webdriver #selenioum webdriver
from selenium.webdriver.common.keys import Keys #selenium input
from selenium.webdriver.common.action_chains import ActionChains #selenium actions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait as wait

chromeDriverPath = "chromedriver.exe"
L_ID = "peanut0223@nate.com"
L_PW = "Jongsul123"

def goPage(driver, urlFullPath):
   try:
      driver.get(urlFullPath)
      driver.implicitly_wait(3)
      return driver.page_source
   except:
      driver.quit()
      return 0

def clickPage(driver, xpath):
   driver.find_element_by_xpath(xpath).click()
   time.sleep(1)

def scrollDown(driver):
   driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
   time.sleep(3)

def putData(driver, user_xpath, user_input):
        pyperclip.copy(user_input) # input을 클립보드로 복사
        driver.find_element_by_xpath(user_xpath).click() # element focus 설정
        ActionChains(driver).key_down(Keys.CONTROL).send_keys('v').key_up(Keys.CONTROL).perform() # Ctrl+V 전달
        time.sleep(1)

# Scenario.s

def A():
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko")
    driver = webdriver.Chrome(chromeDriverPath, chrome_options=options)
    goPage(driver, "https://www.libreview.com")
    clickPage(driver, '//*[@id="country-select"]')
    clickPage(driver, '//*[@id="country-select"]/option[42]')
    clickPage(driver, '//*[@id="submit-button"]')
    time.sleep(3)
    putData(driver, '//*[@id="loginForm-email-input"]', L_ID)
    putData(driver, '//*[@id="loginForm-password-input"]', L_PW)
    clickPage(driver, '//*[@id="loginForm-submit-button"]')
    time.sleep(3)
    clickPage(driver, '//*[@id="main-header-reports-nav-link"]/div')
    time.sleep(3)
    clickPage(driver, '//*[@id="pastGlucoseCard-report-button"]')
    time.sleep(40)
    clickPage(driver, '//*[@id="reports-print-button"]')
    print("---------------")
    time.sleep(30)
    

    
    

A()
