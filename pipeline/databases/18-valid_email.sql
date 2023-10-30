-- Resets the attribute valid_email only when the email has been changed
delimiter //
DROP TRIGGER IF EXISTS valid_email;
CREATE TRIGGER valid_email
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF old.email != new.email THEN
        SET new.valid_email = 0;
    ELSE
        SET new.valid_email = 1;
    END IF;
END;//
delimiter ;