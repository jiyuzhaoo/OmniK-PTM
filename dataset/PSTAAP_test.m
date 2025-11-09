%function [PSTAAP1,PSTAAP2,PSTAAP3,PSTAAP4,PSTAAP5,PSTAAP6,PSTAAP7,PSTAAP8,PSTAAP9,PSTAAP10,PSTAAP11]=PSTAAP()
clear;
clc;
AA='ACDEFGHIKLMNPQRSTVWY';

%1
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b11=[b2,b3,b4,b5,b6,b7,b8,b9,b10,b1_1];
M1=length(b1{1,1});
M11=length(b11{1,1});
x1=(M1+1)/2;
x11=(M11+1)/2;

n1=length(b1);
for i1=1:n1
   b1{1,i1}(x1)='';
end
n11=length(b11);
for i11=1:n11
   b11{1,i11}(x11)='';
end

M1=length(b1{1,1});
M11=length(b11{1,1});
F1=zeros(20^3,M1-2);
F11=zeros(20^3,M11-2);
for m1=1:n1
    for j1=1:M1-2
        t1=b1{1,m1}(j1);
        k1=strfind(AA,t1);
        t11=b1{1,m1}(j1+1);
        k11=strfind(AA,t11);
        t12=b1{1,m1}(j1+2);
        k12=strfind(AA,t12);
        F1(400*(k1-1)+20*(k11-1)+k12,j1)=F1(400*(k1-1)+20*(k11-1)+k12,j1)+1;
    end
end
F1=F1/n1;
for m11=1:n11
    for j11=1:M11-2
        t1=b11{1,m11}(j11);
        k1=strfind(AA,t1);
        t11=b11{1,m11}(j11+1);
        k11=strfind(AA,t11);
        t12=b11{1,m11}(j11+2);
        k12=strfind(AA,t12);
        F11(400*(k1-1)+20*(k11-1)+k12,j11)=F11(400*(k1-1)+20*(k11-1)+k12,j11)+1;
    end
end
F11=F11/n11;

%2
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b21=[b1,b3,b4,b5,b6,b7,b8,b9,b10,b1_1];
M2=length(b2{1,1});
M21=length(b21{1,1});
x2=(M2+1)/2;
x21=(M21+1)/2;

n2=length(b2);
for i2=1:n2
   b2{1,i2}(x2)='';
end
n21=length(b21);
for i21=1:n21
   b21{1,i21}(x21)='';
end

M2=length(b2{1,1});
M21=length(b21{1,1});
F2=zeros(20^3,M2-2);
F21=zeros(20^3,M21-2);
for m2=1:n2
    for j2=1:M2-2
        t2=b2{1,m2}(j2);
        k2=strfind(AA,t2);
        t21=b2{1,m2}(j2+1);
        k21=strfind(AA,t21);
        t22=b2{1,m2}(j2+2);
        k22=strfind(AA,t22);
        F2(400*(k2-1)+20*(k21-1)+k22,j2)=F2(400*(k2-1)+20*(k21-1)+k22,j2)+1;
    end
end
F2=F2/n2;
for m21=1:n21
    for j21=1:M21-2
        t2=b21{1,m21}(j21);
        k2=strfind(AA,t2);
        t21=b21{1,m21}(j21+1);
        k21=strfind(AA,t21);
        t22=b21{1,m21}(j21+2);
        k22=strfind(AA,t22);
        F21(400*(k2-1)+20*(k21-1)+k22,j21)=F21(400*(k2-1)+20*(k21-1)+k22,j21)+1;
    end
end
F21=F21/n21;

%3
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b31=[b1,b2,b4,b5,b6,b7,b8,b9,b10,b1_1];
M3=length(b3{1,1});
M31=length(b31{1,1});
x3=(M3+1)/2;
x31=(M31+1)/2;

n3=length(b3);
for i3=1:n3
   b3{1,i3}(x3)='';
end
n31=length(b31);
for i31=1:n31
   b31{1,i31}(x31)='';
end
M3=length(b3{1,1});
M31=length(b31{1,1});
F3=zeros(20^3,M3-2);
F31=zeros(20^3,M31-2);
for m3=1:n3
    for j3=1:M3-2
        t3=b3{1,m3}(j3);
        k3=strfind(AA,t3);
        t31=b3{1,m3}(j3+1);
        k31=strfind(AA,t31);
        t32=b3{1,m3}(j3+2);
        k32=strfind(AA,t32);
        F3(400*(k3-1)+20*(k31-1)+k32,j3)=F3(400*(k3-1)+20*(k31-1)+k32,j3)+1;
    end
end
F3=F3/n3;
for m31=1:n31
    for j31=1:M31-2
        t3=b31{1,m31}(j31);
        k3=strfind(AA,t3);
        t31=b31{1,m31}(j31+1);
        k31=strfind(AA,t31);
        t32=b31{1,m31}(j31+2);
        k32=strfind(AA,t32);
        F31(400*(k3-1)+20*(k31-1)+k32,j31)=F31(400*(k3-1)+20*(k31-1)+k32,j31)+1;
    end
end
F31=F31/n31;

%4
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b41=[b1,b2,b3,b5,b6,b7,b8,b9,b10,b1_1];
M4=length(b4{1,1});
M41=length(b41{1,1});
x4=(M4+1)/2;
x41=(M41+1)/2;

n4=length(b4);
for i4=1:n4
   b4{1,i4}(x4)='';
end
n41=length(b41);
for i41=1:n41
   b41{1,i41}(x41)='';
end
M4=length(b4{1,1});
M41=length(b41{1,1});
F4=zeros(20^3,M4-2);
F41=zeros(20^3,M41-2);
for m4=1:n4
    for j4=1:M4-2
        t4=b4{1,m4}(j4);
        k4=strfind(AA,t4);
        t41=b4{1,m4}(j4+1);
        k41=strfind(AA,t41);
        t42=b4{1,m4}(j4+2);
        k42=strfind(AA,t42);
        F4(400*(k4-1)+20*(k41-1)+k42,j4)=F4(400*(k4-1)+20*(k41-1)+k42,j4)+1;
    end
end
F4=F4/n4;
for m41=1:n41
    for j41=1:M41-2
        t4=b41{1,m41}(j41);
        k4=strfind(AA,t4);
        t41=b41{1,m41}(j41+1);
        k41=strfind(AA,t41);
        t42=b41{1,m41}(j41+2);
        k42=strfind(AA,t42);
        F41(400*(k4-1)+20*(k41-1)+k42,j41)=F41(400*(k4-1)+20*(k41-1)+k42,j41)+1;
    end
end
F41=F41/n41;

%5
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b51=[b1,b2,b3,b4,b6,b7,b8,b9,b10,b1_1];
M5=length(b5{1,1});
M51=length(b51{1,1});
x5=(M5+1)/2;
x51=(M51+1)/2;

n5=length(b5);
for i5=1:n5
   b5{1,i5}(x5)='';
end
n51=length(b51);
for i51=1:n51
   b51{1,i51}(x51)='';
end

M5=length(b5{1,1});
M51=length(b51{1,1});
F5=zeros(20^3,M5-2);
F51=zeros(20^3,M51-2);
for m5=1:n5
    for j5=1:M5-2
        t5=b5{1,m5}(j5);
        k5=strfind(AA,t5);
        t51=b5{1,m5}(j5+1);
        k51=strfind(AA,t51);
        t52=b5{1,m5}(j5+2);
        k52=strfind(AA,t52);
        F5(400*(k5-1)+20*(k51-1)+k52,j5)=F5(400*(k5-1)+20*(k51-1)+k52,j5)+1;
    end
end
F5=F5/n5;
for m51=1:n51
    for j51=1:M51-2
        t5=b51{1,m51}(j51);
        k5=strfind(AA,t5);
        t51=b51{1,m51}(j51+1);
        k51=strfind(AA,t51);
        t52=b51{1,m51}(j51+2);
        k52=strfind(AA,t52);
        F51(400*(k5-1)+20*(k51-1)+k52,j51)=F51(400*(k5-1)+20*(k51-1)+k52,j51)+1;
    end
end
F51=F51/n51;

%6
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b61=[b1,b2,b3,b4,b5,b7,b8,b9,b10,b1_1];
M6=length(b6{1,1});
M61=length(b61{1,1});
x6=(M6+1)/2;
x61=(M61+1)/2;

n6=length(b6);
for i6=1:n6
   b6{1,i6}(x6)='';
end
n61=length(b61);
for i61=1:n61
   b61{1,i61}(x61)='';
end

M6=length(b6{1,1});
M61=length(b61{1,1});
F6=zeros(20^3,M6-2);
F61=zeros(20^3,M61-2);
for m6=1:n6
    for j6=1:M6-2
        t6=b6{1,m6}(j6);
        k6=strfind(AA,t6);
        t61=b6{1,m6}(j6+1);
        k61=strfind(AA,t61);
        t62=b6{1,m6}(j6+2);
        k62=strfind(AA,t62);
        F6(400*(k6-1)+20*(k61-1)+k62,j6)=F6(400*(k6-1)+20*(k61-1)+k62,j6)+1;
    end
end
F6=F6/n6;
for m61=1:n61
    for j61=1:M61-2
        t6=b61{1,m61}(j61);
        k6=strfind(AA,t6);
        t61=b61{1,m61}(j61+1);
        k61=strfind(AA,t61);
        t62=b61{1,m61}(j61+2);
        k62=strfind(AA,t62);
        F61(400*(k6-1)+20*(k61-1)+k62,j61)=F61(400*(k6-1)+20*(k61-1)+k62,j61)+1;
    end
end
F61=F61/n61;

%7
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b71=[b1,b2,b3,b4,b5,b6,b8,b9,b10,b1_1];
M7=length(b7{1,1});
M71=length(b71{1,1});
x7=(M7+1)/2;
x71=(M71+1)/2;

n7=length(b7);
for i7=1:n7
   b7{1,i7}(x7)='';
end
n71=length(b71);
for i71=1:n71
   b71{1,i71}(x71)='';
end

M7=length(b7{1,1});
M71=length(b71{1,1});
F7=zeros(20^3,M7-2);
F71=zeros(20^3,M71-2);
for m7=1:n7
    for j7=1:M7-2
        t7=b7{1,m7}(j7);
        k7=strfind(AA,t7);
        t71=b7{1,m7}(j7+1);
        k71=strfind(AA,t71);
        t72=b7{1,m7}(j7+2);
        k72=strfind(AA,t72);
        F7(400*(k7-1)+20*(k71-1)+k72,j7)=F7(400*(k7-1)+20*(k71-1)+k72,j7)+1;
    end
end
F7=F7/n7;
for m71=1:n71
    for j71=1:M71-2
        t7=b71{1,m71}(j71);
        k7=strfind(AA,t7);
        t71=b71{1,m71}(j71+1);
        k71=strfind(AA,t71);
        t72=b71{1,m71}(j71+2);
        k72=strfind(AA,t72);
        F71(400*(k7-1)+20*(k71-1)+k72,j71)=F71(400*(k7-1)+20*(k71-1)+k72,j71)+1;
    end
end
F71=F71/n71;
%8
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b81=[b1,b2,b3,b4,b5,b6,b7,b9,b10,b1_1];
M8=length(b8{1,1});
M81=length(b81{1,1});
x8=(M8+1)/2;
x81=(M81+1)/2;
n8=length(b8);
for i8=1:n8
   b8{1,i8}(x8)='';
end
n81=length(b81);
for i81=1:n81
   b81{1,i81}(x81)='';
end

M8=length(b8{1,1});
M81=length(b81{1,1});
F8=zeros(20^3,M8-2);
F81=zeros(20^3,M81-2);
for m8=1:n8
    for j8=1:M8-2
        t8=b8{1,m8}(j8);
        k8=strfind(AA,t8);
        t81=b8{1,m8}(j8+1);
        k81=strfind(AA,t81);
        t82=b8{1,m8}(j8+2);
        k82=strfind(AA,t82);
        F8(400*(k8-1)+20*(k81-1)+k82,j8)=F8(400*(k8-1)+20*(k81-1)+k82,j8)+1;
    end
end
F8=F8/n8;
for m81=1:n81
    for j81=1:M81-2
        t8=b81{1,m81}(j81);
        k8=strfind(AA,t8);
        t81=b81{1,m81}(j81+1);
        k81=strfind(AA,t81);
        t82=b81{1,m81}(j81+2);
        k82=strfind(AA,t82);
        F81(400*(k8-1)+20*(k81-1)+k82,j81)=F81(400*(k8-1)+20*(k81-1)+k82,j81)+1;
    end
end
F81=F81/n81;
%9
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b91=[b1,b2,b3,b4,b5,b6,b7,b8,b10,b1_1];
M9=length(b9{1,1});
M91=length(b91{1,1});
x9=(M9+1)/2;
x91=(M91+1)/2;
n9=length(b9);
for i9=1:n9
   b9{1,i9}(x9)='';
end
n91=length(b91);
for i91=1:n91
   b91{1,i91}(x91)='';
end

M9=length(b9{1,1});
M91=length(b91{1,1});
F9=zeros(20^3,M9-2);
F91=zeros(20^3,M91-2);
for m9=1:n9
    for j9=1:M9-2
        t9=b9{1,m9}(j9);
        k9=strfind(AA,t9);
        t91=b9{1,m9}(j9+1);
        k91=strfind(AA,t91);
        t92=b9{1,m9}(j9+2);
        k92=strfind(AA,t92);
        F9(400*(k9-1)+20*(k91-1)+k92,j9)=F9(400*(k9-1)+20*(k91-1)+k92,j9)+1;
    end
end
F9=F9/n9;
for m91=1:n91
    for j91=1:M91-2
        t9=b91{1,m91}(j91);
        k9=strfind(AA,t9);
        t91=b91{1,m91}(j91+1);
        k91=strfind(AA,t91);
        t92=b91{1,m91}(j91+2);
        k92=strfind(AA,t92);
        F91(400*(k9-1)+20*(k91-1)+k92,j91)=F91(400*(k9-1)+20*(k91-1)+k92,j91)+1;
    end
end
F91=F91/n91;
%10
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b101=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b1_1];
M10=length(b10{1,1});
M101=length(b101{1,1});
x10=(M10+1)/2;
x101=(M101+1)/2;
n10=length(b10);
for i10=1:n10
   b10{1,i10}(x10)='';
end
n101=length(b101);
for i101=1:n101
   b101{1,i101}(x101)='';
end

M10=length(b10{1,1});
M101=length(b101{1,1});
F10=zeros(20^3,M10-2);
F101=zeros(20^3,M101-2);
for m10=1:n10
    for j10=1:M10-2
        t10=b10{1,m10}(j10);
        k10=strfind(AA,t10);
        t101=b10{1,m10}(j10+1);
        k101=strfind(AA,t101);
        t102=b10{1,m10}(j10+2);
        k102=strfind(AA,t102);
        F10(400*(k10-1)+20*(k101-1)+k102,j10)=F10(400*(k10-1)+20*(k101-1)+k102,j10)+1;
    end
end
F10=F10/n10;
for m101=1:n101
    for j101=1:M101-2
        t10=b101{1,m101}(j101);
        k10=strfind(AA,t10);
        t101=b101{1,m101}(j101+1);
        k101=strfind(AA,t101);
        t102=b101{1,m101}(j101+2);
        k102=strfind(AA,t102);
        F101(400*(k10-1)+20*(k101-1)+k102,j101)=F101(400*(k10-1)+20*(k101-1)+k102,j101)+1;
    end
end
F101=F101/n101;
%11
[a1 b1]=fastaread('(1)Test4062.txt');
[a2 b2]=fastaread('(2)Test304.txt');
[a3 b3]=fastaread('(3)Test257.txt');
[a4 b4]=fastaread('(4)Test194.txt');
[a5,b5]=fastaread('(5)Test240.txt');
[a6 b6]=fastaread('(6)Test107.txt');
[a7 b7]=fastaread('(7)Test154.txt');
[a8 b8]=fastaread('(8)Test42.txt');
[a9 b9]=fastaread('(9)Test72.txt');
[a10 b10]=fastaread('(10)Test194.txt');
[a1_1 b1_1]=fastaread('(11)Test36.txt');
b111=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10];
M1_1=length(b1_1{1,1});
M111=length(b111{1,1});
x1_1=(M1_1+1)/2;
x111=(M111+1)/2;
n1_1=length(b1_1);
for i1_1=1:n1_1
   b1_1{1,i1_1}(x1_1)='';
end
n111=length(b111);
for i111=1:n111
   b111{1,i111}(x111)='';
end

M1_1=length(b1_1{1,1});
M111=length(b111{1,1});
F1_1=zeros(20^3,M1_1-2);
F111=zeros(20^3,M111-2);
for m1_1=1:n1_1
    for j1_1=1:M1_1-2
        t1_1=b1_1{1,m1_1}(j1_1);
        k1_1=strfind(AA,t1_1);
        t111=b1_1{1,m1_1}(j1_1+1);
        k111=strfind(AA,t111);
        t112=b1_1{1,m1_1}(j1_1+2);
        k112=strfind(AA,t112);
        F1_1(400*(k1_1-1)+20*(k111-1)+k112,j1_1)=F1_1(400*(k1_1-1)+20*(k111-1)+k112,j1_1)+1;
    end
end
F1_1=F1_1/n1_1;
for m111=1:n111
    for j111=1:M111-2
        t1_1=b111{1,m111}(j111);
        k1_1=strfind(AA,t1_1);
        t111=b111{1,m111}(j111+1);
        k111=strfind(AA,t111);
        t112=b111{1,m111}(j111+2);
        k112=strfind(AA,t112);
        F111(400*(k1_1-1)+20*(k111-1)+k112,j111)=F111(400*(k1_1-1)+20*(k111-1)+k112,j111)+1;
    end
end
F111=F111/n111;

%%%%%%%%%%%%Construct feature vectors
F=(F1+F2+F3+F4+F5+F6+F7+F8+F9+F10+F1_1)/11;
FF=(F11+F21+F31+F41+F51+F61+F71+F81+F91+F101+F111)/11;
Fr=F-FF;
PSTAAP1=zeros(n1,M1-2); 
for m1=1:n1
    for j1=1:M1-2
        t1=b1{1,m1}(j1);
        k1=strfind(AA,t1);
        t11=b1{1,m1}(j1+1);
        k11=strfind(AA,t11);
        t12=b1{1,m1}(j1+2);
        k12=strfind(AA,t12);
        PSTAAP1(m1,j1)=Fr(400*(k1-1)+20*(k11-1)+k12,j1);
    end
end
PSTAAP2=zeros(n2,M2-2); 
for m2=1:n2
    for j2=1:M2-2
        t2=b2{1,m2}(j2);
        k2=strfind(AA,t2);
        t21=b2{1,m2}(j2+1);
        k21=strfind(AA,t21);
        t22=b2{1,m2}(j2+2);
        k22=strfind(AA,t22);
        PSTAAP2(m2,j2)=Fr(400*(k2-1)+20*(k21-1)+k22,j2);
    end
end
PSTAAP3=zeros(n3,M3-2); 
for m3=1:n3
    for j3=1:M3-2
        t3=b3{1,m3}(j3);
        k3=strfind(AA,t3);
        t31=b3{1,m3}(j3+1);
        k31=strfind(AA,t31);
        t32=b3{1,m3}(j3+2);
        k32=strfind(AA,t32);
        PSTAAP3(m3,j3)=Fr(400*(k3-1)+20*(k31-1)+k32,j3);
    end
end
PSTAAP4=zeros(n4,M4-2); 
for m4=1:n4
    for j4=1:M4-2
        t4=b4{1,m4}(j4);
        k4=strfind(AA,t4);
        t41=b4{1,m4}(j4+1);
        k41=strfind(AA,t41);
        t42=b4{1,m4}(j4+2);
        k42=strfind(AA,t42);
        PSTAAP4(m4,j4)=Fr(400*(k4-1)+20*(k41-1)+k42,j4);
    end
end
PSTAAP5=zeros(n5,M5-2); 
for m5=1:n5
    for j5=1:M5-2
        t5=b5{1,m5}(j5);
        k5=strfind(AA,t5);
        t51=b5{1,m5}(j5+1);
        k51=strfind(AA,t51);
        t52=b5{1,m5}(j5+2);
        k52=strfind(AA,t52);
        PSTAAP5(m5,j5)=Fr(400*(k5-1)+20*(k51-1)+k52,j5);
    end
end
PSTAAP6=zeros(n6,M6-2); 
for m6=1:n6
    for j6=1:M6-2
        t6=b6{1,m6}(j6);
        k6=strfind(AA,t6);
        t61=b6{1,m6}(j6+1);
        k61=strfind(AA,t61);
        t62=b6{1,m6}(j6+2);
        k62=strfind(AA,t62);
        PSTAAP6(m6,j6)=Fr(400*(k6-1)+20*(k61-1)+k62,j6);
    end
end
PSTAAP7=zeros(n7,M7-2); 
for m7=1:n7
    for j7=1:M7-2
        t7=b7{1,m7}(j7);
        k7=strfind(AA,t7);
        t71=b7{1,m7}(j7+1);
        k71=strfind(AA,t71);
        t72=b7{1,m7}(j7+2);
        k72=strfind(AA,t72);
        PSTAAP7(m7,j7)=Fr(400*(k7-1)+20*(k71-1)+k72,j7);
    end
end
PSTAAP8=zeros(n8,M8-2); 
for m8=1:n8
    for j8=1:M8-2
        t8=b8{1,m8}(j8);
        k8=strfind(AA,t8);
        t81=b8{1,m8}(j8+1);
        k81=strfind(AA,t81);
        t82=b8{1,m8}(j8+2);
        k82=strfind(AA,t82);
        PSTAAP8(m8,j8)=Fr(400*(k8-1)+20*(k81-1)+k82,j8);
    end
end
PSTAAP9=zeros(n9,M9-2); 
for m9=1:n9
    for j9=1:M9-2
        t9=b9{1,m9}(j9);
        k9=strfind(AA,t9);
        t91=b9{1,m9}(j9+1);
        k91=strfind(AA,t91);
        t92=b9{1,m9}(j9+2);
        k92=strfind(AA,t92);
        PSTAAP9(m9,j9)=Fr(400*(k9-1)+20*(k91-1)+k92,j9);
    end
end
PSTAAP10=zeros(n10,M10-2); 
for m10=1:n10
    for j10=1:M10-2
        t10=b10{1,m10}(j10);
        k10=strfind(AA,t10);
        t101=b10{1,m10}(j10+1);
        k101=strfind(AA,t101);
        t102=b10{1,m10}(j10+2);
        k102=strfind(AA,t102);
        PSTAAP10(m10,j10)=Fr(400*(k10-1)+20*(k101-1)+k102,j10);
    end
end
PSTAAP11=zeros(n1_1,M1_1-2); 
for m1_1=1:n1_1
    for j1_1=1:M1_1-2
        t1_1=b1_1{1,m1_1}(j1_1);
        k1_1=strfind(AA,t1_1);
        t111=b1_1{1,m1_1}(j1_1+1);
        k111=strfind(AA,t111);
        t112=b1_1{1,m1_1}(j1_1+2);
        k112=strfind(AA,t112);
        PSTAAP11(m1_1,j1_1)=Fr(400*(k1_1-1)+20*(k111-1)+k112,j1_1);
    end
end
%save("Fr_test.mat","Fr")
save("PSTAAP_test.mat","PSTAAP1","PSTAAP2","PSTAAP3","PSTAAP4","PSTAAP5","PSTAAP6","PSTAAP7","PSTAAP8","PSTAAP9","PSTAAP10","PSTAAP11")