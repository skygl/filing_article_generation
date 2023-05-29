import re

from html_table_extractor.extractor import Extractor


def table_to_text(table):
    table = f'''{table}'''
    extractor = Extractor(table)
    extractor.parse()
    content_list = extractor.return_list()

    tmp = []
    for content in content_list:
        for c in content:
            c = c.replace("\xa0", "")
            if c not in tmp:
                tmp.append(c)
            else:
                if c == '-':
                    tmp.append(c)
                else:
                    tmp.append('')
        tmp.append('\n')  # 행 구분 개행문자 추가

    result = ','.join(tmp)
    result = re.sub(r'(-,)+', "-,", result)  # -,-,-, 중복 제거

    return result
