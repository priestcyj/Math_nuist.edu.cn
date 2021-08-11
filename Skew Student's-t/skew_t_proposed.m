
clc
clear


gama=1;
zata=3;
Beta=20;
   
        
% Initial image            
image1=imread(['simulated brain images\194_N3F40.bmp']);
[row,column]=size(image1);
% ground truth
image123=imread(['simulated brain images\ground194.bmp']);
           
image1=double(image1);
im1=double(image1);
image123=double(image123);
im1=im1/max(im1(:));
Y=[im1(:)'];
%%-初始化参数
tic
K=3;%类别
[n,m]=size(Y);%n维
Img=reshape(Y,[row,column]);
ROI=ones(row,column);
ROI(Img==0)=0;
index = find(ROI ==1);

%---初始化sigma-----%
yy = Img(index);
yy = sort(yy,'descend');
% K-means for initialization
[IDX,C] = kmeans(yy,K,'Start','cluster', ...
    'Maxiter',100, ...
    'EmptyAction','drop', ...
    'Display','off');
while sum(isnan(C))>0
    [IDX,C] = kmeans(yy,K,'Start','cluster', ...
        'Maxiter',100, ...
        'EmptyAction','drop', ...
        'Display','off');
end
V = sort(C);
Dis_k = zeros(row,column,K);
for k = 1:K
    Dis_k(:,:,k) = (Img - V(k)).^2;
end
Sigma=zeros(K,1);
Pi=ones(m,K)/K;
delta=zeros(K,n);
Lambda=zeros(K,n);
Delta=zeros(K,n);
Gamma=zeros(K,n);
for k = 1:K
    [e_min,IDX] = min(Dis_k,[],3);
    IDX_ROI = IDX.*ROI;
    Sigma(k) = var(Img(IDX_ROI==k));
    Pi(:,k)=repmat(sum(sum(double(IDX_ROI==k)))/(sum(ROI(:))+eps),m,1);
    delta(k)=Lambda(k)/(sqrt(1+Lambda(k)'*Lambda(k))+eps);
    Delta(k)=sqrt(Sigma(k))*delta(k);
    Gamma(k)=Sigma(k)-Delta(k)*Delta(k)';
end
Mu=V;%mu初始化
%-------------------%

%---------初始化G_j------------%

N=4;
M=N*(N+1)/2;%中共多找个基函数
G_j=zeros(M,m);%得到的基函数
bias=zeros(row,column,M);%得到的基函数
    X=linspace(-1,1,row);
    P(1,1:row)=1;
    P(2,1:row)=X;
    P(3,1:row)=1/2*(3*X.^2-1);
    P(4,1:row)=1/2*(5*X.^3-3*X);
    P(5,1:row)=1/8*(35*X.^4-30*X.^2+3);
    P(6,1:row)=1/8*(63*X.^5-70*X.^3+15*X);
    P(7,1:row)=1/16*(231*X.^6-315*X.^4+105*X.^2-5);
    P(8,1:row)=1/16*(429*X.^7-693*X.^5+315*X.^3-35.*X);

    X=linspace(-1,1,column);
    Q(1,1:column)=1;
    Q(2,1:column)=X;
    Q(3,1:column)=1/2*(3*X.^2-1);
    Q(4,1:column)=1/2*(5*X.^3-3*X);
    Q(5,1:column)=1/8*(35*X.^4-30*X.^2+3);
    Q(6,1:column)=1/8*(63*X.^5-70*X.^3+15*X);
    Q(7,1:column)=1/16*(231*X.^6-315*X.^4+105*X.^2-5);
    Q(8,1:column)=1/16*(429*X.^7-693*X.^5+315*X.^3-35.*X);

    num=1;
    for i=1:N
        for j=1:N-i+1
            for ki=1:row
                for kj=1:column
                    bias(ki,kj,num)=P(i,ki)*Q(j,kj);
                end
            end
            num=num+1;
        end
    end
    for i=1:M
        temp=bias(:,:,i);
        G_j(i,:)=temp(:)';
    end
%---------初始化B_j------------%
B_j=ones(1,m);
S1_ij=zeros(K,m);S2_ij=zeros(K,m);
v=5*ones(K,1);%自由度是5；
ROI=ROI(:); 
patch_size=3;

%% 计算权重系数 alpha_ij
[alpha_ij,ci]=cacluate_alpha_ij(im1./(reshape(B_j,[row,column])+eps),m,row,column);


for t=1:50
    
    
    %----------------计算Z_ij,u_ij,u_ij*t_ij,u_ij*t_ij*t_ij，logu_ij------------------------------%
    Mi=(1+Delta.*(Gamma.^(-1)).*Delta).^(-1/2);
    d=(repmat(Y,K,1)-repmat(B_j,K,1).*repmat(Mu,1,m)).^2.*repmat((Sigma.^(-1)),1,m);
    tD=repmat(gamma((v+n)/2),1,m).*repmat((Sigma.^(-1/2)),1,m)...
        ./(repmat((pi*v).^(n/2),1,m).*repmat(gamma(v/2),1,m)+eps)...
        .*((1+d./repmat(v,1,m)).^(-repmat((v+n)/2,1,m)));
    A=repmat(Lambda,1,m).*repmat((Sigma.^(-1/2)),1,m).*(repmat(Y,K,1)-repmat(B_j,K,1).*repmat(Mu,1,m));
    T_2=tcdf(sqrt(repmat(v+n+2,1,m)./(repmat(v,1,m)+d+eps)).*A,repmat(v+n+2,1,m));
    T_1=tcdf(sqrt(repmat(v+n,1,m)./(repmat(v,1,m)+d+eps)).*A,repmat(v+n,1,m));
    u_ij=2*repmat(gamma((v+n+2)/2),1,m)./(repmat(gamma((v+n)/2),1,m).*(repmat(v,1,m)+d)+eps).*T_2./(T_1+eps);
    GV=mufun_gv(d,v,n,K,m,A); 
    logu_ij=u_ij-log((repmat(v,1,m)+d)/2)-repmat(v+n,1,m)./(repmat(v,1,m)+d+eps)+psi(repmat(v/2+n/2,1,m))...
        +tpdf(sqrt(repmat(v+n,1,m)./(repmat(v,1,m)+d+eps)).*A,repmat(v+n,1,m))./(T_1+eps).*(A.*(d-n)./(sqrt(repmat(v+n,1,m).*((repmat(v,1,m)+d).^3))+eps))...
        +1./(T_1+eps).*GV;
%    
   
    S1_ij=repmat(Mi.^2.*Delta,1,m).*repmat((Gamma.^(-1)),1,m).*(repmat(Y,K,1)-repmat(B_j,K,1).*repmat(Mu,1,m)); 
    S2_ij=repmat(gamma((v+n+1)/2),1,m).*((repmat(v,1,m)+d).^((repmat(v,1,m)+n)/2))...
        ./(sqrt(pi)*T_1.*repmat(gamma((v+n)/2),1,m).*(repmat(v,1,m)+d+A.^2).^(repmat((v+n+1)/2,1,m))+eps);
    ut_ij=u_ij.*S1_ij+repmat(Mi,1,m).*S2_ij;
    ut2_ij=u_ij.*S1_ij.^2+repmat(Mi.^2,1,m)+repmat(Mi,1,m).*S1_ij.*S2_ij;
    Z=2*Pi'.*tD.*T_1;
    Z=Z./(repmat(sum(Z),K,1)+eps).*repmat(ROI',K,1);
    z1_ik=Z';%后验概率
 
    %------------------------------更新参数--------------------------%
    for k=1:K
    z1_ik_reshape=reshape(z1_ik(:,k),[row,column]);  
    Pi_reshape=reshape(Pi(:,k),[row,column]);
    A1=[zeros(row,1),z1_ik_reshape,zeros(row,1)];
            B1=[zeros(1,column+2);A1;zeros(1,column+2)];
            A2=[zeros(row,1),Pi_reshape,zeros(row,1)];
            B2=[zeros(1,column+2);A2;zeros(1,column+2)];
            z1n_ik(:,k)=(sum(alpha_ij'.*im2col(B1, [3,3], 'sliding')))';
            Pin(:,k)=(sum(alpha_ij'.*im2col(B2, [3,3], 'sliding')))';
    end
    
    Fik=exp(gama*Pin+zata*z1n_ik)./(repmat(exp(ci),1,K)+eps);
    Fik=Fik./(repmat(sum(Fik,2),1,K)+eps).*repmat(ROI,1,K);
    Pi=Beta*Fik+z1_ik;
    Pi=Pi./(repmat(sum(Pi,2),1,K)+eps).*repmat(ROI,1,K);
    %------------------------------更新Mu----------------------------%
    Mu_old=Mu;
    Mu=sum(Z.*u_ij.*repmat(B_j,K,1).*repmat(Y,K,1)-Z.*ut_ij.*repmat(B_j,K,1).*repmat(Delta,1,m),2)./(sum(Z.*u_ij.*repmat(B_j.^2,K,1),2)+eps);
    Delta=sum(Z.*ut_ij.*(repmat(Y,K,1)-repmat(B_j,K,1).*repmat(Mu,1,m)),2)./(sum(Z.*ut2_ij,2)+eps);
    Gamma=sum((Z.*u_ij.*(repmat(Y,K,1)-repmat(B_j,K,1).*repmat(Mu,1,m)).^2)...
        -2*(Z.*ut_ij.*(repmat(Y,K,1)-repmat(B_j,K,1).*repmat(Mu,1,m)).*repmat(Delta,1,m))...
        +(Z.*ut2_ij.*repmat((Delta).^2,1,m)),2)...
        ./(sum(Z,2)+eps);
    Sigma=Gamma+(Delta.*Delta);
    Lambda=(Sigma.^(-1/2)).*Delta./(sqrt(1-Delta.*(Sigma.^(-1)).*Delta)+eps);
   

    %----------------------------------------------------------------%   
    
      
    
    %------------------------------更新v----------------------------%
    gx=1.0+log(v/2)+sum(Z.*(logu_ij-u_ij),2)./(sum(Z,2)+eps)-psi(v/2);
    gx1=1./v-0.5*psi(1,v/2);
    v=v-gx./(gx1+eps);
    

    
    %----------------------------------------------------------------% 
    
%    disp(sqrt(sum((Mu_old-Mu).^2)))
    if sqrt(sum((Mu_old-Mu).^2)) < 0.001%max(abs(Mu_old-Mu)) < 0.0002
         break;
    end
    
    
    %------------------------------更新W---------------------------%
    
    J1=sum(Z.*u_ij.*repmat(Mu,1,m).*repmat(Gamma.^(-1),1,m).*repmat(Mu,1,m));
    J2=sum(Z.*u_ij.*repmat(Y,K,1).*repmat(Gamma.^(-1),1,m).*repmat(Mu,1,m)-Z.*ut_ij.*repmat(Delta,1,m).*repmat(Gamma.^(-1),1,m).*repmat(Mu,1,m));
    V=sum(G_j.*repmat(J2,M,1),2);
    A=(G_j.*repmat(J1,M,1))*(G_j)';
    W=(A^(-1))*V;    
    B_j=W'*G_j;
    B_j=B_j.*ROI';
    

    
    [~,nn]=max(Z);
      [~,wz]=sort(Mu);
      nn=reshape(nn,[row,column]);
      out2=nn;
      for i=1:K
        out2(nn==wz(i))=50*(i);
      end      
 imshow(out2,[]);
 title(t)
 pause(0.1)  
end
[~,nn]=max(Z);
      [~,wz]=sort(Mu);
      nn=reshape(nn,[row,column]);
      out2=nn;
      for i=1:K
        out2(nn==wz(i))=50*(i);
      end      

    
B=out2;
B=B.*reshape(ROI,[row,column]);
   
    subplot(1,2,1)
    imshow(image123,[])
    colormap(gray);
    subplot(1,2,2)
    imshow(B,[])
    pause(0.1)
      
      
      
      

cc1=0;cc_1=0;cc_2=0;cc_3=0;cc2=0;cc3=0;

[row,column]=size(image123);
for i=1:row
    for j=1:column
        if image123(i,j)==50||B(i,j)==50
            cc1=cc1+1;
        end        
        if image123(i,j)==50&&B(i,j)==50
            cc_1=cc_1+1;
        end  
        if image123(i,j)==100||B(i,j)==100
            cc2=cc2+1;
        end
        if image123(i,j)==100&&B(i,j)==100
            cc_2=cc_2+1;
        end
        
        if image123(i,j)==150||B(i,j)==150
            cc3=cc3+1;
        end
        
        if image123(i,j)==150&&B(i,j)==150
            cc_3=cc_3+1;
        end
        
    end
end

c1_cor=cc_1/(cc1)*100;
c2_cor=cc_2/(cc2)*100;
c3_cor=cc_3/(cc3)*100;
Correct=[c3_cor,c2_cor,c1_cor];

c=0;ccc=0;
for i=1:row
    for j=1:column  
            if (image123(i,j)-B(i,j))==0
                c=c+1;
            end
    end
end
for i=1:row
    for j=1:column
        
            if image123(i,j)==0
            ccc=ccc+1;
            else 
                ccc=ccc;
            end
       
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
correct_total=(c-ccc)/((row*column-ccc)*2-(c-ccc))*100; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

restore_img=double(image1)./(reshape(B_j,[row,column])+eps);

WM_m=mean(restore_img(image123==150));
WM_std=std(restore_img(image123==150));

GM_m=mean(restore_img(image123==100));
GM_std=std(restore_img(image123==100));
cv_GM=GM_std/(GM_m+eps);
cv_WM=WM_std/(WM_m+eps);
cjv=(WM_std+GM_std)/(abs(WM_m-GM_m)+eps);

Data=[Correct,correct_total,cv_WM,cv_GM,cjv];



seg_result=B;
seg_result_bias=reshape(B_j,[row,column]);

% imwrite(uint8(seg_result),['result_true\proposed\IBSR_',num2str(it1),'\IBSR_',num2str(it1),'_ana_strip_',num2str(it),'_new.bmp']); 
% imwrite((seg_result_bias),['result_true\proposed\IBSR_',num2str(it1),'\IBSR_',num2str(it1),'_ana_strip_',num2str(it),'_new_bias.bmp']);
% imwrite(uint8(restore_img),['result_true\proposed\IBSR_',num2str(it1),'\IBSR_',num2str(it1),'_ana_strip_',num2str(it),'_new_restored.bmp']);  

function [GV1]=mufun_gv(d,v,n,K,m,A)
%% 计算积分
Q1=zeros(K,m);
C=sqrt(repmat(v+n,1,m)./(repmat(v,1,m)+d+eps)).*A;
for k=1:K 
fun=@(x)((psi((v(k)+2*n)/2)-psi((v(k)+n)/2)-log(1+((x+C(k,:)).^2)./(v(k)+n+eps))+((x+C(k,:)).^2.*(v(k)+n)-n*(v(k)+n))./(((x+C(k,:)).^2+v(k)+n).*(v(k)+n)+eps)).*tpdf((x+C(k,:)),v(k)+n));
Q1(k,:)=integral(fun,-inf,0,'ArrayValued',true);
end

GV1=Q1;
% GV2=Q2;
end


function [alpha_ij,ci]=cacluate_alpha_ij(im1,m,row,column)
%% 计算权重系数 alpha_ij

Ni=9;
radiu=fix(sqrt(Ni)/2);
index=1;
index_im1=reshape(1:m,[row,column]);
for j=radiu:-1:-radiu
    for i=radiu:-1:-radiu
        temp1=circshift(im1,[i,j]);
        temp2=circshift(index_im1,[i,j]);
        im1_Ni(:,index)=temp1(:);
        index_im1_Ni(:,index)=temp2(:);
        index=index+1;
    end
end
%----------------------w_ij------------------------%
j_label=(Ni+1)-(1:Ni);
center_label=[];
label=index_im1_Ni(index_im1_Ni,j_label);
for j=1:Ni
    center_label=[center_label;label((j-1)*m+1:j*m,j)];
end
wij=abs(im1_Ni(index_im1_Ni,:)-im1_Ni(center_label,:));
wij=sum(wij.^2,2);
wij=reshape(exp(-sqrt(wij)/0.5),[m Ni]);
temp=wij(:,5);
temp=double(double((wij(:,5)./(sum(wij,2)+eps))>=0.5)==0).*temp;
wij(:,5)=temp;
wij=wij./(repmat(sum(wij,2),1,Ni)+eps);
%-----------------------------------------------------------------%    

%----------------------S_i^*------------------------%

%---------eight direction-------------------%
D1=[1,1,1;0,0,0;0,0,0];D2=[0,0,0;1,1,1;0,0,0];D3=[0,0,0;0,0,0;1,1,1];
D4=[0,1,0;0,1,0;0,1,0];D5=[1,0,0;1,0,0;1,0,0];D6=[0,0,1;0,0,1;0,0,1];
D7=[0,0,1;0,1,0;1,0,0];D8=[1,0,0;0,1,0;0,0,1];
D=[D1(:),D2(:),D3(:),D4(:),D5(:),D6(:),D7(:),D8(:)];
d1=im1_Ni'.*repmat(D1(:),1,m);d2=im1_Ni'.*repmat(D2(:),1,m);d3=im1_Ni'.*repmat(D3(:),1,m);d4=im1_Ni'.*repmat(D4(:),1,m);
d5=im1_Ni'.*repmat(D5(:),1,m);d6=im1_Ni'.*repmat(D6(:),1,m);d7=im1_Ni'.*repmat(D7(:),1,m);d8=im1_Ni'.*repmat(D8(:),1,m);
Std=[std(d1([1,4,7],:));std(d2([2,5,8],:));std(d3([3,6,9],:));std(d4([4,5,6],:));std(d5([1,2,3],:));std(d6([7,8,9],:));std(d7([3,5,7],:));std(d3([1,5,9],:))];
[Std_sort,Std_index]=sort(Std,1,'ascend'); 
temp_1=D(:,Std_index(1,:));
temp_2=D(:,Std_index(1,:))+D(:,Std_index(2,:));
temp_3=D(:,Std_index(1,:))+D(:,Std_index(2,:))+D(:,Std_index(3,:));

index_1=find(abs(Std_sort(2,:)-Std_sort(1,:))>=max(diff(Std_sort)));
index_2_1=find(abs(Std_sort(2,:)-Std_sort(1,:))<=max(diff(Std_sort)));
index_2_2=find(abs(Std_sort(3,:)-Std_sort(2,:))>=max(diff(Std_sort)));
index_2=intersect(index_2_1,index_2_2);

S_i=temp_3;
% S_i(:,index_1)=temp_1(:,index_1);
% S_i(:,index_2)=temp_2(:,index_2);
S_i(:,index_1)=double(temp_1(:,index_1)>0);
S_i(:,index_2)=double(temp_2(:,index_2)>0);

temp=S_i;
temp(temp==0)=NaN;
ci=std(temp.*im1_Ni','omitnan')';
alpha_ij=S_i'.*wij;
alpha_ij=alpha_ij./(repmat(sum(alpha_ij,2),1,Ni)+eps);
end