-- the customer table, the constraint is name not null. mobile phone and emial is unique and email has format.
CREATE TABLE customer(
	customer_id varchar(10),
	first_name varchar(20) NOT NULL,
	last_name varchar(20) NOT NULL,
	mobile_number varchar(15) UNIQUE,
	email varchar(50) UNIQUE,
	country varchar(15),
	state varchar(15),
	city varchar(10),
	suburb varchar(10),
	detail_address varchar(100),
	driver_licence varchar(10) UNIQUE,

	PRIMARY KEY (customer_id),
	CHECK (email LIKE '%@%')
);

-- salesperson table, the constraint is name is not null, mobile phone and email is not null, email should have format, salary > 0, commision_rate in (0,10%).
CREATE TABLE salesperson(
	salesperson_id varchar(10),
	salary integer,
	commission_rate float,
	full_name varchar(20) NOT NULL,
	email varchar(50) UNIQUE,
	mobile_number varchar(15) UNIQUE,

	PRIMARY KEY (salesperson_id),
	CHECK (salary > 0),
	CHECK (commission_rate > 0 AND commission_rate < 0.1),
	CHECK (email LIKE '%@%')
);

-- testdrive table, the date, time is not null, and salesperson_id, customer_id is FK.
CREATE TABLE testdrive(
	test_id varchar(10),
	date date NOT NULL,
	time time NOT NULL,
	salesperson_id varchar(10),
	customer_id varchar(10),

	PRIMARY KEY(test_id),
	FOREIGN KEY (salesperson_id) REFERENCES salesperson(salesperson_id) ON DELETE CASCADE,
	FOREIGN KEY (customer_id) REFERENCES customer(customer_id) ON DELETE CASCADE
);

-- table vehicle, the detailed information is not null, and mileage >= 0, vehicle status has 2 status, listed_price should more than 0.
CREATE TABLE vehicle(
	VIN varchar(20) PRIMARY KEY,
	make varchar(20) NOT NULL,
	model varchar(20) NOT NULL,
	build_year integer NOT NULL,
	mileage integer NOT NULL,
	colour varchar(20) NOT NULL,
	transmission_type varchar(20) NOT NULL,
	listed_price integer,
	vehicle_status varchar(20),

	CHECK (mileage >= 0),
	CHECK (vehicle_status IN ('Have been sold', 'For Sale')),
	CHECK (listed_price > 0)	
);

-- the previous owner of the used vehicle.
CREATE TABLE pre_owner(
	pre_owner_id varchar(20),
	pre_owner_name varchar(20) NOT NULL,

	PRIMARY KEY(pre_owner_id)
);

-- table used vehcile, which is a 'ISA' relationship to previous owner. System record the basic information, and several constraint.
CREATE TABLE pre_owned_vehicle(
	VIN varchar(20) PRIMARY KEY,
	mechanical_condition varchar(20),
	body_condition varchar(20),
	value integer,
	trade_in_date date NOT NULL,
	pre_owner_id varchar(20),
	
	FOREIGN KEY (VIN) REFERENCES vehicle(VIN) ON DELETE CASCADE,
	FOREIGN KEY (pre_owner_id) REFERENCES pre_owner(pre_owner_id) ON DELETE SET NULL,
	CHECK (mechanical_condition IN ('poor', 'fair', 'good','excellent')),
	CHECK (body_condition IN ('poor','fair','good','excellent'))
	
);

-- table new_vehicle, which has nothing new attribute compare to vehicle, but for the scalability, we put it in a new table.
CREATE TABLE new_vehicle(
	VIN varchar(20) PRIMARY KEY,
	FOREIGN KEY (VIN) REFERENCES vehicle(VIN) ON DELETE CASCADE
);

-- The detailed_information for a vehicle, which is a week entity to vehicle, so we used mutilple PK
CREATE TABLE detailed_information(
	detailed_id varchar(10),
	VIN varchar(20),
	img_path varchar(100),
	description varchar(200),
	FOREIGN KEY (VIN) REFERENCES vehicle(VIN) ON DELETE CASCADE,
	PRIMARY KEY (VIN, detailed_id)
);

-- the likelihood in the testdrive, where the user can test serval vehicle in 1 testdrive. The information about the VIN of vehicle and feedback from testdrive and user is recorded
CREATE TABLE likelihood(
	VIN varchar(20),
	test_id varchar(10),
	feedback varchar(300),

	PRIMARY KEY (VIN,test_id), -- due to 1 to 1 relationship between VIN and testdrive, we can use mutilple PK.
	FOREIGN KEY (VIN) REFERENCES vehicle(VIN) ON DELETE CASCADE,
	FOREIGN KEY (test_id) REFERENCES testDrive(test_id) ON DELETE CASCADE
);

-- the table about purchasement, including the different entity involved and also some constraint. To make sure that one person can join in 1 purchasement in 1 day, use unique constraint to do it.
CREATE TABLE purchasement(
	purchase_id varchar(10) PRIMARY KEY,
	purchase_status varchar(10),
	salesperson_id varchar(10),
	customer_id varchar(10),
	discount float, 
	base_price integer,
	trade_in_vehicle_id varchar(20),
	date date NOT NULL,
	VIN varchar(20) NULL,

	FOREIGN KEY (trade_in_vehicle_id) REFERENCES pre_owned_vehicle(VIN) ON DELETE SET NULL,
	FOREIGN KEY (VIN) REFERENCES vehicle(VIN) ON DELETE SET NULL,	
	FOREIGN KEY (salesperson_id) REFERENCES salesperson(salesperson_id) ON DELETE SET NULL,
	FOREIGN KEY (customer_id) REFERENCES customer(customer_id) ON DELETE SET NULL,
	CHECK (base_price > 0),
	CHECK (discount > 0 and discount <= 1),
	CHECK (purchase_status IN ('pending', 'completed')),

	UNIQUE(date, customer_id)

);

-- the aftermarket option is to record serval aftermarket option available.
CREATE TABLE aftermarket(
	aftermarket_id varchar(10) PRIMARY KEY,
	aftermarket_name varchar(20) NOT NULL,
	aftermarket_description varchar(200)
);

-- after purchase table is to record the option user choose in 1 purchasement.
CREATE TABLE after_purchase(
	after_purchase_id varchar(10) PRIMARY KEY,
	purchase_id varchar(10),
	aftermarket_id varchar(10),
	price integer,

	CHECK(price > 0),
	FOREIGN KEY (purchase_id) REFERENCES purchasement(purchase_id) ON DELETE CASCADE,
	FOREIGN KEY (aftermarket_id) REFERENCES aftermarket(aftermarket_id) ON DELETE CASCADE
);

-- payment is for purchasement, which is a weak entity. The constraint is as below, pay_option is in a range, and pay_amount > 0.
CREATE TABLE payment(
	purchase_id varchar(10),
	pay_id varchar(10) UNIQUE,
	pay_options varchar(20),
	pay_amount integer,
	PRIMARY KEY (purchase_id, pay_id),
	FOREIGN KEY (purchase_id) REFERENCES purchasement(purchase_id) ON DELETE CASCADE,
	CHECK (pay_options IN ('cash', 'credit card', 'bank transfer', 'bank financing')),
	CHECK (pay_amount > 0)
);

-- bank_finacing is for the payment. The constraint is loan term >= 12 and <= 50.
CREATE TABLE bank_financing(
  bank_financing_id varchar(10) PRIMARY KEY,
  pay_id varchar(10),
  loan_amount integer,
  bank_unikey varchar(20),
  loan_term integer,
  interest_range float,
  proof_document_unikey varchar(20),
  application_date date NOT NULL, 
  CHECK (loan_term >= 12 AND loan_term <= 50),
  FOREIGN KEY (pay_id) REFERENCES payment(pay_id) ON DELETE CASCADE

);

-- trigger, check when user want to add a new aftermarket option, whether it is a new vehicle, if not, raise exception.
CREATE OR REPLACE FUNCTION checkAftermarketForNewVehicle() RETURNS trigger AS $$
DECLARE
  v_vin VARCHAR(20);
BEGIN
  SELECT VIN INTO v_vin FROM purchasement WHERE purchase_id = NEW.purchase_id;

  IF NOT EXISTS (
    SELECT 1 FROM new_vehicle WHERE VIN = v_vin
  ) THEN
    RAISE EXCEPTION 'Aftermarket options can only be added to new vehicles.';
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_checkAftermarket
BEFORE INSERT ON after_purchase
FOR EACH ROW
EXECUTE FUNCTION checkAftermarketForNewVehicle();