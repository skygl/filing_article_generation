import re

from html_table_extractor.extractor import Extractor


def table_to_text(table):
    table = f'''{table}'''
    extractor = Extractor(table)
    extractor.parse()
    content_list = extractor.return_list()

    total = ''
    prev_tmp = []
    for content in content_list:
        tmp = []
        tmp_str = []
        for c_idx, c in enumerate(content):
            c = c.replace("\xa0", "")
            if c not in tmp:
                if len(prev_tmp) > c_idx and c == prev_tmp[c_idx]:
                    tmp_str.append("")
                else:
                    tmp_str.append(c)
            else:
                if c == '-':
                    tmp_str.append(c)
                else:
                    tmp_str.append("")
            tmp.append(c)

        tmp.append('\n')  # 행 구분 개행문자 추가
        tmp_str = "\t".join(tmp_str) + "\n"
        total += tmp_str
        prev_tmp = tmp

    result = re.sub(r'(—)+', "-", total)
    result = "\n" + result

    return result
