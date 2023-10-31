-- -- Initial
-- DROP TABLE IF EXISTS corrections;
-- DROP TABLE IF EXISTS users;
-- DROP TABLE IF EXISTS projects;

-- CREATE TABLE IF NOT EXISTS users (
--     id int not null AUTO_INCREMENT,
--     name varchar(255) not null,
--     average_score float default 0,
--     PRIMARY KEY (id)
-- );

-- CREATE TABLE IF NOT EXISTS projects (
--     id int not null AUTO_INCREMENT,
--     name varchar(255) not null,
--     PRIMARY KEY (id)
-- );

-- CREATE TABLE IF NOT EXISTS corrections (
--     user_id int not null,
--     project_id int not null,
--     score int default 0,
--     KEY `user_id` (`user_id`),
--     KEY `project_id` (`project_id`),
--     CONSTRAINT fk_user_id FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
--     CONSTRAINT fk_project_id FOREIGN KEY (`project_id`) REFERENCES `projects` (`id`) ON DELETE CASCADE
-- );

-- INSERT INTO users (name) VALUES ("Bob");
-- SET @user_bob = LAST_INSERT_ID();

-- INSERT INTO users (name) VALUES ("Jeanne");
-- SET @user_jeanne = LAST_INSERT_ID();

-- INSERT INTO projects (name) VALUES ("C is fun");
-- SET @project_c = LAST_INSERT_ID();

-- INSERT INTO projects (name) VALUES ("Python is cool");
-- SET @project_py = LAST_INSERT_ID();


-- INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_c, 80);
-- INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_py, 96);

-- INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_c, 91);
-- INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_py, 73);


-- Initial
DROP TABLE IF EXISTS corrections;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS projects;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    average_score float default 0,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS projects (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS corrections (
    user_id int not null,
    project_id int not null,
    score int default 0,
    KEY `user_id` (`user_id`),
    KEY `project_id` (`project_id`),
    CONSTRAINT fk_user_id FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
    CONSTRAINT fk_project_id FOREIGN KEY (`project_id`) REFERENCES `projects` (`id`) ON DELETE CASCADE
);

INSERT INTO users (name) VALUES ("user_0");
SET @user_0 = LAST_INSERT_ID();
INSERT INTO users (name) VALUES ("user_1");
SET @user_1 = LAST_INSERT_ID();
INSERT INTO users (name) VALUES ("user_2");
SET @user_2 = LAST_INSERT_ID();
INSERT INTO users (name) VALUES ("user_3");
SET @user_3 = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("project_0");
SET @project_0 = LAST_INSERT_ID();
INSERT INTO projects (name) VALUES ("project_1");
SET @project_1 = LAST_INSERT_ID();
INSERT INTO projects (name) VALUES ("project_2");
SET @project_2 = LAST_INSERT_ID();
INSERT INTO projects (name) VALUES ("project_3");
SET @project_3 = LAST_INSERT_ID();


INSERT INTO corrections (user_id, project_id, score) VALUES (@user_0, @project_0, 10);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_1, @project_0, 20);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_1, @project_1, 30);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_2, @project_0, 40);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_2, @project_1, 50);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_2, @project_2, 60);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_3, @project_0, 70);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_3, @project_1, 80);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_3, @project_2, 90);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_3, @project_3, 10);