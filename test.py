from src.data_util import read_file
import re


def get_word_index(word, index, sentence):
    location = sentence.find(word)
    if location < 0:
        location = index + len(sentence) - 1
    return location


def resort_tags():
    file_path = "/Users/zxj/result_20170321_20171231_video_seg_for_train"
    file_iter = read_file(file_path,
                          pre_process=lambda x: x.strip().split("\t"))
    file_iter = (line[2:] for line in file_iter if len(line) == 5)
    output_path = "/Users/zxj/Downloads/result_20170321_20171231_tag_sorted"

    common_words = {"国际", "国内", "华语", "流行", "欧美"}
    with open(output_path, mode="w+", encoding="utf-8") as file:
        for tags, sentence, words in file_iter:
            tag_list = tags.split("$$")
            tag_list = (tag for tag in tag_list if tag not in common_words)
            tag_list = [(tag, get_word_index(tag, index, sentence)) for
                        index, tag in enumerate(tag_list)]
            tag_list = sorted(tag_list, key=lambda x: x[1])
            tag_list = [tag for tag, _ in tag_list]
            new_tags = "$$".join(tag_list)
            result = new_tags + "\t" + sentence + "\t" + words + "\n"
            file.write(result)


if __name__ == '__main__':
    file_path = "/Users/zxj/Downloads/result_20170321_20171231_tag_sorted"
    output_path = "/Users/zxj/Downloads/result_20170321_20171231_filtered"
    file_iter = read_file(file_path,
                          pre_process=lambda x: x.strip().split("\t"))
    file_iter = filter(lambda x: len(x) == 3, file_iter)
    common_words = {"内地", "歪果仁"}
    misleading_tags = {"ios", "编程", "黑马程序员"}
    with open(output_path, mode="w+", encoding="utf-8") as file:
        for tags, sentence, words in file_iter:
            tag_list = tags.split("$$")
            word_list = words.split("\001")
            new_word_list = [word.split("\002")[0] for word in word_list]
            should_filter = len(new_word_list) >= 30
            new_tag_list = []
            for tag in tag_list:
                if tag in misleading_tags:
                    should_filter = True
                    break
                if tag not in common_words:
                    new_tag_list.append(tag)

            if should_filter:
                continue
            new_tags = "$$".join(new_tag_list)
            new_words = "\001".join(new_word_list)
            result = new_tags + "\t" + new_words + "\n"
            file.write(result)