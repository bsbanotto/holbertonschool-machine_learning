-- Creates a table `users` with the following attributes
-- id: integer, never NULL, auto increment, is primary key
-- email: string, 255 characters, never NULL is unique
-- name: string, 255 characters
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(256)
);