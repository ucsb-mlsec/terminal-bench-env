datatype Point = Point(x: real, y: real)

function DistanceSquared(p1: Point, p2: Point): real
{
    (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)
}

method ComputeDistance(p1: Point, p2: Point) returns (distSq: real)
    ensures distSq == DistanceSquared(p1, p2)
    ensures distSq >= 0.0
{
    var dx := p1.x - p2.x;
    var dy := p1.y - p2.y;
    distSq := dx * dx + dy * dy;
}

function CrossProduct(p1: Point, p2: Point, p3: Point): real
{
    (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
}

method IsLeftTurn(p1: Point, p2: Point, p3: Point) returns (leftTurn: bool)
    ensures leftTurn <==> CrossProduct(p1, p2, p3) > 0.0
{
    var cross := CrossProduct(p1, p2, p3);
    leftTurn := cross > 0.0;
}

method Midpoint(p1: Point, p2: Point) returns (mid: Point)
    ensures mid.x == (p1.x + p2.x) / 2.0
    ensures mid.y == (p1.y + p2.y) / 2.0
    ensures 2.0 * DistanceSquared(p1, mid) == DistanceSquared(p1, p2)
    ensures 2.0 * DistanceSquared(p2, mid) == DistanceSquared(p1, p2)
{
    mid := Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
}

method TriangleInequality(p1: Point, p2: Point, p3: Point) returns (valid: bool)
    requires p1 != p2 && p2 != p3 && p1 != p3
    ensures valid ==> (
        DistanceSquared(p1, p2) > 0.0 &&
        DistanceSquared(p2, p3) > 0.0 &&
        DistanceSquared(p1, p3) > 0.0
    )
    ensures valid ==> (
        var d12 := DistanceSquared(p1, p2);
        var d23 := DistanceSquared(p2, p3);
        var d13 := DistanceSquared(p1, p3);
        var s12 := if d12 >= 0.0 then d12 else 0.0;
        var s23 := if d23 >= 0.0 then d23 else 0.0;
        var s13 := if d13 >= 0.0 then d13 else 0.0;
        (s12 + s23 >= s13 - 0.0001) && (s12 + s13 >= s23 - 0.0001) && (s23 + s13 >= s12 - 0.0001)
    )
{
    var d12 := DistanceSquared(p1, p2);
    var d23 := DistanceSquared(p2, p3);
    var d13 := DistanceSquared(p1, p3);
    
    var dx12 := p1.x - p2.x;
    var dy12 := p1.y - p2.y;
    var dx23 := p2.x - p3.x;
    var dy23 := p2.y - p3.y;
    var dx13 := p1.x - p3.x;
    var dy13 := p1.y - p3.y;
    
    var s12 := dx12 * dx12 + dy12 * dy12;
    var s23 := dx23 * dx23 + dy23 * dy23;
    var s13 := dx13 * dx13 + dy13 * dy13;
    
    var area_component := (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    var area_sq := area_component * area_component;
    
    valid := (d12 > 0.0) && (d23 > 0.0) && (d13 > 0.0) &&
             (s12 + s23 >= s13 - 0.0001) && (s12 + s13 >= s23 - 0.0001) && (s23 + s13 >= s12 - 0.0001);
}

method CheckCollinear(p1: Point, p2: Point, p3: Point) returns (collinear: bool)
    ensures collinear <==> (CrossProduct(p1, p2, p3) == 0.0)
{
    var cross := CrossProduct(p1, p2, p3);
    collinear := cross == 0.0;
}