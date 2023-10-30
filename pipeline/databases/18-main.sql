-- Show users and update (or not) email
SELECT * FROM users;

UPDATE users SET email = "email3@test.com" WHERE email = "email3@test.com";

SELECT "--";
SELECT * FROM users;

UPDATE users SET name = "New name" WHERE email = "email3@test.com";

SELECT "--";
SELECT * FROM users;