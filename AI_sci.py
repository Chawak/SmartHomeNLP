import re 
from pythainlp.util import text_to_num,num_to_thaiword
from pythainlp.tokenize import word_tokenize
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from googletrans import Translator
from sentence_transformers import SentenceTransformer
import pickle


class Solution():
    
    def __init__(self,room_list=["ห้องครัว","ห้องนอน","ห้องโถง","ห้องน้ำ"]):
        
        self.idx2command = ["invalid","turn on","turn off"]
        self.room_list = room_list
        
        self.classifier_feature_extractor = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.classifier_filename = "lr_eng.pkl"
        self.classifier = pickle.load(open(self.classifier_filename, 'rb'))
        
        self.translator = Translator()
        
        self.qa_threshold = 1e-6
        self.qa_model_name = "deepset/roberta-base-squad2"        

        self.qa_model = pipeline('question-answering', model=self.qa_model_name, tokenizer=self.qa_model_name)
        
        self.place_qa_template = [0,{'question':'Where to turn on the light ?'},{'question':'Where to turn off the light ?'}]
        self.time_qa_template = [0,{'question':'When to turn on the light ?'},{'question':'When to turn off the light ?'}]
    
    def rule_based_classify(self,text):
        words = word_tokenize(text)

        word2labels = {"ปิดไฟ":2,"เปิดไฟ":1,"มืด":1}
        for word in word2labels:
            if word in words:
                return word2labels[word]

        if "ปิด" in words and "ไฟ" in words:
            return 2

        if "เปิด" in words and "ไฟ" in words:
            return 1
        if "สว่าง" in words and "ขึ้น" in words:
            return 1
        return 0
    
    def rule_based_extract_time(self,text):

        tee = [1,2,3,4,5]
        bai = [1,2,3,4,5]
        tum = [1,2,3,4,5]
        mong = [2,3,4,5,6,7,8,9,10,11]
        clock = [i for i in range(1,25)]
        clock_format = ["\d+[\.|:]\d\d"]

        tee += [num_to_thaiword(elem) for elem in tee]
        tee = ["ตี\s*"+str(elem) for elem in tee] 
        
        bai += [num_to_thaiword(elem) for elem in bai]
        bai = ["บ่าย\s*"+str(elem) for elem in bai] 
        
        tum += [num_to_thaiword(elem) for elem in tum]
        tum = [str(elem)+"\s*ทุ่ม" for elem in tum] 

        mong += [num_to_thaiword(elem) for elem in mong]
        mong = [str(elem)+"\s*โมง" for elem in mong] 
        
        clock += [num_to_thaiword(elem) for elem in clock]
        clock = [str(elem)+"\s*นาฬิกา" for elem in clock] 

        others = ["เที่ยง","เที่ยงคืน","ตอนเย็น","ตอนเช้า","ตอนกลางวัน","ตอนดึก","ตอนกลางคืน","พรุ่งนี้","มะรืนนี้","วันต่อไป"]
        
        time_text= re.findall("|".join(clock_format+others+tum+tee+bai+mong+clock),text)
        
        
        num_sentence = [tok for tok in text_to_num(text) if tok not in [''," "]]
        
        

        for i,word in enumerate(num_sentence[:-1]):
            if num_sentence[i+1] in ["วินาที","นาที","ชั่วโมง","วัน","เดือน","ปี"]:
                try:
                    float_num = float(word)
                    time_text.append(word+" "+num_sentence[i+1])
                except:
                    pass
            
        return " ".join(time_text)
    
    
    def solve(self,text,engine="rule-based"):

        if engine=="rule-based":
            command = self.rule_based_classify(text)
            
            if command == 0 : 
                return {"command":"invalid command"}
            
            place = []
            for room in self.room_list:
                if room in text:
                    place.append(room)
            place=",".join(place)

            time = self.rule_based_extract_time(text)

            ans = {"command":self.idx2command[command],"device":"lights","room":place,"time":time}
            return ans
        
        elif engine=="classify-QA":

            translated_text = self.translator.translate([text])[0].text

            command = self.classifier.predict(self.classifier_feature_extractor.encode([translated_text]))[0]

            if command == 0 : 
                return {"command":"invalid command"}

            self.place_qa_template[command]["context"] = translated_text
            place_prediction = self.qa_model(self.place_qa_template[command])

            if place_prediction["score"]<self.qa_threshold:
                place = ""
            else:
                place = place_prediction["answer"]


            self.time_qa_template[command]["context"] = translated_text
            time_prediction = self.qa_model(self.time_qa_template[command])

            if time_prediction["score"]<self.qa_threshold:
                time = ""
            else:
                time = time_prediction["answer"]

            # print(place_prediction,time_prediction)            
            ans = {"command":self.idx2command[command],"device":"lights","room":place,"time":time}
            return ans

solution = Solution()
text_list = ["ปิดไฟห้องน้ำให้หน่อย",
             "เปิดไฟห้องน้ำให้หน่อย",
             "เปิดไฟในห้องน้ำให้ทีตอน 17.00",
             "ปิดไฟตอนเที่ยงคืน",
             "อยากกินไก่ KFC",
             "ปิดไฟในครัว",
             "ปิดไฟในห้องนอนตอนสิบเจ็ดนาฬิกา",
             "ปิดไฟในห้องนอนตอน 17 นาฬิกา",
             "เปิดไฟในอีก 2 ชั่วโมง 17 นาที 20 วินาที",
             "เปิดไฟในช่วง 3 ทุ่มถึง 5 ทุ่ม"]

ava_text_list = ["เปิดไฟห้องนอนหน่อย","ปิดไฟโถงล่างตอนสามทุ่ม","เปิดไฟซีนเอ"]

for text in text_list:
    print(text)
    print(solution.solve(text,engine="rule-based"))
    print(solution.solve(text,engine="classify-QA"))
        
    