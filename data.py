from pathlib import Path
import json
def process_data(data,result=[]):
    if isinstance(data,(dict,list)):
        if isinstance(data,(dict)) :
            for key,value in data.items():
                if key == "Content" and isinstance(value,str):
                    result.append(data["Content"])
        for key, content in data.items() if isinstance(data,dict) else enumerate(data):
            if isinstance(content,(dict,list,str)):
                process_data(content,result)
    elif isinstance(data,str):
        pass
    with open("sample.json", "w") as outfile:
        json.dump(result, outfile)

    return result

def jsontotext(data, result=[]):
    if isinstance(data, (dict, list)):
        for key, content in data.items() if isinstance(data, dict) else enumerate(data):
            if isinstance(content, (dict, list)):
                jsontotext(content, result)
            elif isinstance(content, str):
                result.append(content)
    return result
