-- Lists all records of second_table greater than 10 sorted by score
SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC;