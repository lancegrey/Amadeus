import itchat


def text_collection(msg):
    pass


itchat.msg_register("text")(text_collection)


if __name__ == "__main__":
    itchat.auto_login()
    itchat.run()

