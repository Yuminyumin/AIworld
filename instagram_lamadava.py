import datetime
import logging
import multiprocessing
import os

from dotenv import load_dotenv
from lamadava import AsyncClient, Client

import CustomErrors
from remove_special_char import remove_special_characters_using_regex
from translation import translate_text


class LamadavaUtils:

    # 로거 설정
    logger = logging.getLogger()

    def __init__(self):
        load_dotenv()
        token = os.environ.get("LAMADAVA_TOKEN")
        # API토큰 사용
        self.as_cl = AsyncClient(token=token)
        self.cl = Client(token=token)

    def post_by_user(self, user_name):
        '''
        인스타그램 id를 통한 최대 30개 게시글 조회 및 번역
        :param user_name:
        :return:
        '''
        try:
            user_info = self.cl.user_by_username_v2(user_name)

            # 계정이 존재하는 경우
            if 'user' in user_info:
                is_private = user_info['user']["is_private"]
                media_count = user_info['user']["media_count"]

                if is_private:
                    raise CustomErrors.PrivateAccountError("비공개 계정입니다.")
                if media_count == 0:
                    raise CustomErrors.NoPostError("게시물이 없습니다.")

                # 최대 조회 게시글 수 30개로 제한
                media_count = 30 if media_count > 30 else media_count
                user_id = int(user_info['user']['pk'])

                container = []  # 게시글 반환 컨테이너
                all_text = ""   # 게시글 텍스트

                try:
                    self.cl.user_medias(user_id, count=media_count, container=container, max_requests=2)
                except Exception as e:
                    raise e

                for media in container:
                    if media['caption'] is not None:
                        # 각 게시글 당 텍스트만 추출
                        post_text = media["caption"]["text"]
                        all_text += post_text

                # 특수기호 및 개행문자 제거
                all_text = remove_special_characters_using_regex(all_text).replace("\n", "")

                # 비 영문 게시글 영어로 번역
                translated_text = translate_text(all_text)

                return [translated_text]

            # 계정이 존재하지 않는 경우
            else:
                raise CustomErrors.NoAccountError("존재하지 않는 계정입니다.")

        except CustomErrors.CustomError as e:
            self.logger.info(e)
            raise e
        except Exception as e:
            self.logger.error(e)
            raise e

    def dict_post_by_user(self, user_name, return_dict):
        text = self.post_by_user(user_name)
        return_dict[user_name] = text

def pr_e(i, t_dict):
    t_dict[i] = i

def post_by(cl, user_name):
    '''
    인스타그램 id를 통한 최대 30개 게시글 조회 및 번역
    :param user_name:
    :return:
    '''
    try:
        print(cl)

        user_info = cl.user_by_username_v2(user_name)
        print(user_info)

        # 계정이 존재하는 경우
        if 'user' in user_info:
            is_private = user_info['user']["is_private"]
            media_count = user_info['user']["media_count"]

            if is_private:
                raise CustomErrors.PrivateAccountError("비공개 계정입니다.")
            if media_count == 0:
                raise CustomErrors.NoPostError("게시물이 없습니다.")

            # 최대 조회 게시글 수 30개로 제한
            media_count = 30 if media_count > 30 else media_count
            user_id = int(user_info['user']['pk'])

            container = []  # 게시글 반환 컨테이너
            all_text = ""   # 게시글 텍스트

            try:
                cl.user_medias(user_id, count=media_count, container=container, max_requests=2)
            except Exception as e:
                raise e

            for media in container:
                if media['caption'] is not None:
                    # 각 게시글 당 텍스트만 추출
                    post_text = media["caption"]["text"]
                    all_text += post_text

            # 특수기호 및 개행문자 제거
            all_text = remove_special_characters_using_regex(all_text).replace("\n", "")

            # 비 영문 게시글 영어로 번역
            translated_text = translate_text(all_text)

            return [translated_text]

        # 계정이 존재하지 않는 경우
        else:
            raise CustomErrors.NoAccountError("존재하지 않는 계정입니다.")

    except CustomErrors.CustomError as e:
        raise e
    except Exception as e:
        raise e

def dict_post(cl, user, t_dict):
    text = post_by(cl, user)
    t_dict[user] = text


if __name__ == '__main__':

    start = datetime.datetime.now()
    id_list = ["dlwlrma", "gym_junseo", "charlieputh", "bbakdo__", "godjp"]  # 인스타그램 id 리스트

    # 1. 단일 실행 (약 33초 소요)
    # util = LamadavaUtils()
    # text_dict = {}
    # for i_id in id_list:
    #     text_dict[i_id] = util.post_by_user(i_id)
    # print(text_dict)

    #2. 멀티프로세싱
    load_dotenv()
    cl_list = []

    token_lamadava = "pMSwvDOTXKvW56yw6hsYrQAEa7WSEfAv"

    for i in range(5):
        t = os.environ.get("LAMADAVA_TOKEN_"+str(i))
        cl_list.append(Client(token=t))

    manager = multiprocessing.Manager()
    text_dict = manager.dict()

    jobs = []
    for i in range(len(id_list)):
        p = multiprocessing.Process(target=dict_post, args=(cl_list[i], id_list[i], text_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for user_id, text in text_dict.items():
        print(f"User: {user_id}\nText: {text}\n")


    print(datetime.datetime.now()-start)
