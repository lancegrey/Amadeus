import itchat
import ChatBot

@itchat.msg_register("text")
def text_reply(msg):
    if msg['FromUserName'] != "":
        ret_massage = ChatBot.replay(msg['Text'])
        return ret_massage


if __name__ == "__main__":
    itchat.auto_login()
    itchat.run()
