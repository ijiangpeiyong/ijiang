#include <iostream>
using namespace std;
struct Foo {
    int n;
    Foo() {
       cout << "static constructor\n";
    }
    ~Foo() {
       cout << "static destructor\n";
    }
};
Foo f; // static object
int main()
{
    cout << "main function\n";
}




