import schedule
import time
from datetime import datetime

def output_file():
	filename = './scheduler_test/{}.txt'.format(datetime.today().strftime('%Y-%m-%d-%H-%M'))
	with open(filename, "w") as text_file:
		text_file.write("write out at date and time: {}".format(datetime.today().strftime('%Y-%m-%d-%H-%M')))



if __name__ == "__main__":
	schedule.every(30).seconds.do(output_file)	

	while True: 
	    # Checks whether a scheduled task  
	    # is pending to run or not 
	    schedule.run_pending() 
	    time.sleep(1)