-- -- Show and compute average score
-- SELECT * FROM users;
-- SELECT * FROM corrections;

-- SELECT "--";
-- CALL ComputeAverageScoreForUser((SELECT id FROM users WHERE name = "Jeanne"));

-- SELECT "--";
-- SELECT * FROM users;


-- Add one bonus of an existing project
SELECT * FROM users;
SELECT * FROM projects;
SELECT * FROM corrections;

SELECT "--";

CALL ComputeAverageScoreForUser((SELECT id FROM users WHERE name = "user_3"));

SELECT "--";

SELECT * FROM users;
SELECT * FROM projects;
SELECT * FROM corrections;