#
# Example Python client for University Registration DB
#

import psycopg2


class PythonClient:
    # connection parameters - ENTER YOUR LOGIN AND PASSWORD HERE
    userid = "y25s1c9120_mzha0323"
    passwd = "zmj20020619"
    myHost = "awsprddbs4836.shared.sydney.edu.au"

    # instance variable for the database connection
    conn = None


    # Establishes a connection to the database.
    # The connection parameters are read from the instance variables above
    # (userid, passwd, and database).
    # @returns  true   on success and then the instance variable 'conn'
    #                  holds an open connection to the database.
    #           false  otherwise
    def connectToDatabase(self):
        try:
            # connect to the database
            self.conn = psycopg2.connect(
                database=self.userid,
                user=self.userid,
                password=self.passwd,
                host=self.myHost,
            )
            return True

        except psycopg2.Error as sqle:
            # TODO: add error handling #
            print("psycopg2.Error : ", str(sqle))
            return False

    # open ONE single database connection
    def openConnection(self):
        retval = True
        if self.conn is not None:
            print(
                "You are already connected to the database no second connection is needed!"
            )
        else:
            if self.connectToDatabase():
                print("You are successfully connected to the database.")
            else:
                print("Oops - something went wrong.")
                retval = False
        return retval

    # close the database connection again
    def closeConnection(self):
        if self.conn is None:
            print("You are not connected to the database!")
        else:
            try:
                self.conn.close()  # close the connection again after usage!
                self.conn = None
            except psycopg2.Error as sqle:
                # TODO: add error handling #
                print("psycopg2.Error : ", str(sqle))


	# Example Function:
	# Lists on the screen all course offerings ascending by uos_Code
	# including all semesters when the course is offered.
    def listUnits(self):

        try:
			 # Assumes that we are already connected to the database
            curs = self.conn.cursor()

            # execute the query
            curs.execute(
                "SELECT uosCode, uosName, credits, semester, year FROM UoSOffering JOIN UnitOfStudy USING (uosCode) ORDER BY uosCode,year,semester"
            )

            #  loop through the resultset
            nr = 0
            row = curs.fetchone()
            while row is not None:
                nr += 1
                print(
                    str(row[0])
                    + " - "
                    + str(row[1])
                    + " ("
                    + str(row[2])
                    + "cp) "
                    + str(row[4])
                    + "-"
                    + str(row[3])
                )
                row = curs.fetchone()

            if nr == 0:
                print("No entries found.")

            # clean up! (NOTE this really belongs in a finally block)
            curs.close()

        except psycopg2.Error as sqle:
            # TODO: add error handling #
            print("psycopg2.Error : ", str(sqle))


##
# Main program.
##

# create our actual client and test the database connection
uniDB = PythonClient()
if uniDB.openConnection():
    # original example function
    print("OK!")
   