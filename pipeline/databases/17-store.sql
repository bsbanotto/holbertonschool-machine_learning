-- Updates quantities in items table when order is entered in orders table
DROP TRIGGER IF EXISTS decrease_item_quantity;
CREATE TRIGGER decrease_item_quantity
AFTER INSERT ON orders
FOR EACH ROW
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE items.name = NEW.item_name;
