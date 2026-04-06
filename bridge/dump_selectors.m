// dump_selectors.m — Dump _ANEInMemoryModel selectors to find macOS 15 API changes
// Compile: xcrun clang -O2 -fobjc-arc -o dump_selectors dump_selectors.m -framework Foundation -ldl
// Run: ./dump_selectors

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <dlfcn.h>

void dump_class_methods(const char *className) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:className]);
    if (!cls) {
        printf("  ❌ Class not found: %s\n", className);
        return;
    }

    printf("\n=== %s instance methods ===\n", className);
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char *name = sel_getName(sel);
        const char *types = method_getTypeEncoding(methods[i]);
        printf("  - %s  (%s)\n", name, types ? types : "?");
    }
    free(methods);
    printf("  Total: %u methods\n", count);

    // Also check class methods
    printf("\n=== %s class methods ===\n", className);
    count = 0;
    methods = class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("  + %s\n", sel_getName(sel));
    }
    free(methods);
    printf("  Total: %u class methods\n", count);
}

int main(void) {
    @autoreleasepool {
        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW);
        if (!handle) {
            printf("Failed to load ANE framework\n");
            return 1;
        }

        printf("ANE Framework loaded successfully\n");
        printf("macOS version: %s\n", [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String]);

        const char *classes[] = {
            "_ANEInMemoryModel",
            "_ANEInMemoryModelDescriptor",
            "_ANERequest",
            "_ANEIOSurfaceObject",
            "_ANEModel",
            "_ANEClient",
            NULL
        };

        for (int i = 0; classes[i]; i++) {
            dump_class_methods(classes[i]);
        }

        return 0;
    }
}
