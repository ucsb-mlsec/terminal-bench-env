/* Simple two-process handshake protocol model */

/* Global state variables */
bool req = false;      /* request flag from process 1 */
bool ack = false;      /* acknowledgment flag from process 2 */
bool grant = false;    /* grant flag from process 2 to process 1 */
byte state1 = 0;       /* state of process 1: 0=idle, 1=requesting, 2=granted, 3=releasing */
byte state2 = 0;       /* state of process 2: 0=idle, 1=processing */
bool critical = false; /* indicates if process 1 is in critical section */

/* Process 1: Requester */
proctype Process1()
{
    do
    :: (state1 == 0) ->
        /* Request access */
        atomic {
            state1 = 1;
            req = true;
        }
        
    :: (state1 == 1 && grant == true) ->
        /* Enter critical section */
        atomic {
            state1 = 2;
            critical = true;
        }
        
    :: (state1 == 2) ->
        /* Exit critical section and release */
        atomic {
            critical = false;
            state1 = 3;
            req = false;
        }
        
    :: (state1 == 3 && ack == false) ->
        /* Return to idle */
        atomic {
            state1 = 0;
            grant = false;
        }
    od
}

/* Process 2: Granter */
proctype Process2()
{
    do
    :: (state2 == 0 && req == true && grant == false) ->
        /* Process request and grant access */
        atomic {
            state2 = 1;
            grant = true;
            ack = false;
        }
        
    :: (state2 == 1 && req == false) ->
        /* Acknowledge release and return to idle */
        atomic {
            ack = true;
            state2 = 0;
        }
        
    :: (state2 == 0 && ack == true && req == false) ->
        /* Reset acknowledgment */
        ack = false;
    od
}

/* Initialize both processes */
init
{
    atomic {
        run Process1();
        run Process2();
    }
}