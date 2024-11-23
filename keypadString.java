    import java.util.*;

class keypadString{
    public static void main(String ar[]){
        HashMap<Integer,String> hm = new HashMap<>();
        int n = 215,r=0;
        String s="";
        hm.put(0," ");
        hm.put(1," ");
        
        hm.put(2,"abc");
        hm.put(3,"def");
        hm.put(4,"ghi");
        hm.put(5,"jkl");
        hm.put(6,"mno");
        hm.put(7,"pqrs");
        hm.put(8,"tuv");
        hm.put(9,"wxyz");

        System.out.println(hm);


        while(n>0){
            r = n%10;
            s = hm.get(r) + s;
            n/=10;
        }
        
        System.out.print(s);
 
    }
}