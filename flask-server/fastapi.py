from fastapi import FastAPI

app = FastAPI()


init = '''CREATE EXTENSION IF NOT EXISTS pgcrypto;
BEGIN; '''

create_tables = '''CREATE TABLE IF NOT EXISTS public.chat_history
(
    chat_history_id bigserial NOT NULL,
    user_id bigserial NOT NULL,
    conversation text COLLATE pg_catalog."default",
    title text COLLATE pg_catalog."default",
    CONSTRAINT chat_history_id PRIMARY KEY (chat_history_id)
); 

CREATE TABLE IF NOT EXISTS public.users
(
    user_id bigint NOT NULL,
    username character varying(50) COLLATE pg_catalog."default" NOT NULL,
    password character varying(255) COLLATE pg_catalog."default" NOT NULL,
    role character varying(50) COLLATE pg_catalog."default" NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_pkey PRIMARY KEY (user_id),
    CONSTRAINT users_username_key UNIQUE (username)
);

CREATE TABLE IF NOT EXISTS public.chat_history_split
(
    chat_history_id bigint NOT NULL,
    user_message text COLLATE pg_catalog."default" NOT NULL,
    chatbot_message text COLLATE pg_catalog."default" NOT NULL,
    date_time date NOT NULL
);  '''

constraints = ''' ALTER TABLE IF EXISTS public.chat_history
    ADD CONSTRAINT user_id FOREIGN KEY (user_id)
    REFERENCES public.users (user_id) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;


ALTER TABLE IF EXISTS public.chat_history_split
    ADD CONSTRAINT chat_history_id FOREIGN KEY (chat_history_id)
    REFERENCES public.chat_history (chat_history_id) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;  '''

insert = ''' insert into users (username, password, role) VALUES
('Bob', crypt('abc123', gen_salt('bf')), 'customer'),
('Alice', crypt('123456', gen_salt('bf')), 'customer'),
('Raymond', crypt('p4ssw0rd', gen_salt('bf')), 'engineer');
'''
end_init = "END;"

@app.get("/")
async def create_tables():
    
