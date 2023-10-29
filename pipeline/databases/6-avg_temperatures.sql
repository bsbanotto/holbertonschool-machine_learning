-- Displays the average temperature by city ordered by temperature (descending)
SELECT city, AVG(value) AS average_temp
FROM temperatures
GROUP BY city
ORDER BY average_temp DESC;
