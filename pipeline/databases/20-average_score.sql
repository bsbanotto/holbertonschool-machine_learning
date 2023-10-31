-- SQL script that creates a stored procedure `ComputeAverageScoreForUser`
-- that computes and stores the average score for a student
delimiter //

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    DECLARE users_average_score DECIMAL(3, 1);

    -- Calculate the average score for the user
    SELECT AVG(score) INTO users_average_score
    FROM corrections
    WHERE corrections.user_id = user_id;

    -- Update the user's average score in the users table
    UPDATE users
    SET users.average_score = users_average_score
    WHERE users.id = user_id;
END;//

delimiter ;