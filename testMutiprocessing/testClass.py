#coding=utf-8

import os
import multiprocessing
from time import ctime, sleep
from selenium import webdriver


class testClass(object):

    def worker(self, interval, browser="Chrome", url="http://loginurl"):
        driver = eval("webdriver.%s()" % browser)
        driver.get(url)
        driver.find_element_by_id("txtUserName").send_keys("username")
        driver.find_element_by_id("txtPwd").send_keys("password")
        sleep(1)
        driver.find_element_by_id("btnLogin").click()
        print("I was at the %s %s" % (browser, ctime()))
        sleep(interval)
        print("end worker_2")


if __name__ == "__main__":
    for i in range(2):
        a = testClass()
        p = multiprocessing.Process(target=a.worker, args=(2, "Chrome"))
        p.start()
        sleep(2)

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
    print("END!!!!!!!!!!!!!!!!!")







