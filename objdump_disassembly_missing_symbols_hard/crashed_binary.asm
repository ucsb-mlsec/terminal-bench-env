crashed_binary.asm

  401000: f3 0f 1e fa           endbr64
  401004: 48 83 ec 08           sub    $0x8,%rsp
  401008: 48 8b 05 d9 2f 00 00  mov    0x2fd9(%rip),%rax
  40100f: 48 85 c0              test   %rax,%rax
  401012: 74 02                 je     401016
  401014: ff d0                 call   *%rax
  401016: 48 83 c4 08           add    $0x8,%rsp
  40101a: c3                    ret
  40101b: 0f 1f 44 00 00        nopl   0x0(%rax,%rax,1)
  401020: f3 0f 1e fa           endbr64
  401024: 48 8d 3d e5 2f 00 00  lea    0x2fe5(%rip),%rdi
  40102b: 48 8d 05 de 2f 00 00  lea    0x2fde(%rip),%rax
  401032: 48 39 f8              cmp    %rdi,%rax
  401035: 74 15                 je     40104c
  401037: 48 8b 05 a2 2f 00 00  mov    0x2fa2(%rip),%rax
  40103e: 48 85 c0              test   %rax,%rax
  401041: 74 09                 je     40104c
  401043: ff e0                 jmp    *%rax
  401045: 0f 1f 00              nopl   (%rax)
  401048: c3                    ret
  401049: 0f 1f 80 00 00 00 00  nopl   0x0(%rax)
  401050: f3 0f 1e fa           endbr64
  401054: c3                    ret
  401055: 66 2e 0f 1f 84 00 00  cs nopw 0x0(%rax,%rax,1)
  40105c: 00 00 00
  40105f: 90                    nop
  401060: f3 0f 1e fa           endbr64
  401064: 31 ed                 xor    %ebp,%ebp
  401066: 49 89 d1              mov    %rdx,%r9
  401069: 5e                    pop    %rsi
  40106a: 48 89 e2              mov    %rsp,%rdx
  40106d: 48 83 e4 f0           and    $0xfffffffffffffff0,%rsp
  401071: 50                    push   %rax
  401072: 54                    push   %rsp
  401073: 45 31 c0              xor    %r8d,%r8d
  401076: 31 c9                 xor    %ecx,%ecx
  401078: 48 8d 3d ca 00 00 00  lea    0xca(%rip),%rdi
  40107f: ff 15 53 2f 00 00     call   *0x2f53(%rip)
  401085: f4                    hlt
  401086: 66 2e 0f 1f 84 00 00  cs nopw 0x0(%rax,%rax,1)
  40108d: 00 00 00
  401090: 48 8d 3d 79 2f 00 00  lea    0x2f79(%rip),%rdi
  401097: 48 8d 35 72 2f 00 00  lea    0x2f72(%rip),%rsi
  40109e: 48 29 fe              sub    %rdi,%rsi
  4010a1: 48 89 f0              mov    %rsi,%rax
  4010a4: 48 c1 ee 3f           shr    $0x3f,%rsi
  4010a8: 48 c1 f8 03           sar    $0x3,%rax
  4010ac: 48 01 c6              add    %rax,%rsi
  4010af: 48 d1 fe              sar    %rsi
  4010b2: 74 14                 je     4010c8
  4010b4: 48 8b 05 35 2f 00 00  mov    0x2f35(%rip),%rax
  4010bb: 48 85 c0              test   %rax,%rax
  4010be: 74 08                 je     4010c8
  4010c0: ff e0                 jmp    *%rax
  4010c2: 66 0f 1f 44 00 00     nopw   0x0(%rax,%rax,1)
  4010c8: c3                    ret
  4010c9: 0f 1f 80 00 00 00 00  nopl   0x0(%rax)
  4010d0: f3 0f 1e fa           endbr64
  4010d4: 80 3d 35 2f 00 00 00  cmpb   $0x0,0x2f35(%rip)
  4010db: 75 2b                 jne    401108
  4010dd: 55                    push   %rbp
  4010de: 48 83 3d 12 2f 00 00  cmpq   $0x0,0x2f12(%rip)
  4010e5: 00
  4010e6: 48 89 e5              mov    %rsp,%rbp
  4010e9: 74 0c                 je     4010f7
  4010eb: 48 8b 3d 16 2f 00 00  mov    0x2f16(%rip),%rdi
  4010f2: e8 09 ff ff ff        call   401000
  4010f7: e8 64 ff ff ff        call   401060
  4010fc: c6 05 0d 2f 00 00 01  movb   $0x1,0x2f0d(%rip)
  401103: 5d                    pop    %rbp
  401104: c3                    ret
  401105: 0f 1f 00              nopl   (%rax)
  401108: c3                    ret
  401109: 0f 1f 80 00 00 00 00  nopl   0x0(%rax)
  401110: f3 0f 1e fa           endbr64
  401114: e9 67 ff ff ff        jmp    401080
  401119: f3 0f 1e fa           endbr64
  40111d: 55                    push   %rbp
  40111e: 48 89 e5              mov    %rsp,%rbp
  401121: 48 83 ec 20           sub    $0x20,%rsp
  401125: 89 7d ec              mov    %edi,-0x14(%rbp)
  401128: 48 89 75 e0           mov    %rsi,-0x20(%rbp)
  40112c: c7 45 fc 00 00 00 00  movl   $0x0,-0x4(%rbp)
  401133: 83 7d ec 01           cmpl   $0x1,-0x14(%rbp)
  401137: 7f 16                 jg     40114f
  401139: 48 8d 05 c4 0e 00 00  lea    0xec4(%rip),%rax
  401140: 48 89 c7              mov    %rax,%rdi
  401143: e8 08 ff ff ff        call   401050
  401148: b8 01 00 00 00        mov    $0x1,%eax
  40114d: eb 4c                 jmp    40119b
  40114f: 48 8b 45 e0           mov    -0x20(%rbp),%rax
  401153: 48 83 c0 08           add    $0x8,%rax
  401157: 48 8b 00              mov    (%rax),%rax
  40115a: 48 89 c7              mov    %rax,%rdi
  40115d: e8 2e 00 00 00        call   401190
  401162: 89 45 fc              mov    %eax,-0x4(%rbp)
  401165: 83 7d fc 00           cmpl   $0x0,-0x4(%rbp)
  401169: 79 16                 jns    401181
  40116b: 48 8d 05 aa 0e 00 00  lea    0xeaa(%rip),%rax
  401172: 48 89 c7              mov    %rax,%rdi
  401175: e8 d6 fe ff ff        call   401050
  40117a: b8 01 00 00 00        mov    $0x1,%eax
  40117f: eb 1a                 jmp    40119b
  401181: 8b 45 fc              mov    -0x4(%rbp),%eax
  401184: 89 c6                 mov    %eax,%esi
  401186: 48 8d 05 a7 0e 00 00  lea    0xea7(%rip),%rax
  40118d: 48 89 c7              mov    %rax,%rdi
  401190: b8 00 00 00 00        mov    $0x0,%eax
  401195: e8 b6 fe ff ff        call   401050
  40119a: 90                    nop
  40119b: c9                    leave
  40119c: c3                    ret
  40119d: 55                    push   %rbp
  40119e: 48 89 e5              mov    %rsp,%rbp
  4011a1: 48 83 ec 20           sub    $0x20,%rsp
  4011a5: 48 89 7d e8           mov    %rdi,-0x18(%rbp)
  4011a9: c7 45 fc 00 00 00 00  movl   $0x0,-0x4(%rbp)
  4011b0: 48 8b 45 e8           mov    -0x18(%rbp),%rax
  4011b4: 48 89 c7              mov    %rax,%rdi
  4011b7: e8 04 00 00 00        call   4011c0
  4011bc: 89 45 fc              mov    %eax,-0x4(%rbp)
  4011bf: 83 7d fc 00           cmpl   $0x0,-0x4(%rbp)
  4011c3: 79 0a                 jns    4011cf
  4011c5: b8 ff ff ff ff        mov    $0xffffffff,%eax
  4011ca: e9 85 00 00 00        jmp    401254
  4011cf: 48 8b 45 e8           mov    -0x18(%rbp),%rax
  4011d3: 48 89 c7              mov    %rax,%rdi
  4011d6: e8 15 00 00 00        call   4011f0
  4011db: 85 c0                 test   %eax,%eax
  4011dd: 75 07                 jne    4011e6
  4011df: b8 00 00 00 00        mov    $0x0,%eax
  4011e4: eb 6e                 jmp    401254
  4011e6: 8b 45 fc              mov    -0x4(%rbp),%eax
  4011e9: 0f af 45 fc           imul   -0x4(%rbp),%eax
  4011ed: eb 65                 jmp    401254
  4011ef: 90                    nop
  4011f0: 55                    push   %rbp
  4011f1: 48 89 e5              mov    %rsp,%rbp
  4011f4: 48 83 ec 20           sub    $0x20,%rsp
  4011f8: 48 89 7d e8           mov    %rdi,-0x18(%rbp)
  4011fc: 48 8b 45 e8           mov    -0x18(%rbp),%rax
  401200: 48 89 c7              mov    %rax,%rdi
  401203: e8 28 00 00 00        call   401230
  401208: 89 45 fc              mov    %eax,-0x4(%rbp)
  40120b: 83 7d fc 00           cmpl   $0x0,-0x4(%rbp)
  40120f: 75 07                 jne    401218
  401211: b8 00 00 00 00        mov    $0x0,%eax
  401216: eb 16                 jmp    40122e
  401218: 83 7d fc 0a           cmpl   $0xa,-0x4(%rbp)
  40121c: 7e 07                 jle    401225
  40121e: b8 01 00 00 00        mov    $0x1,%eax
  401223: eb 09                 jmp    40122e
  401225: b8 00 00 00 00        mov    $0x0,%eax
  40122a: eb 02                 jmp    40122e
  40122c: eb 00                 jmp    40122e
  40122e: c9                    leave
  40122f: c3                    ret
  401230: 55                    push   %rbp
  401231: 48 89 e5              mov    %rsp,%rbp
  401234: 48 83 ec 20           sub    $0x20,%rsp
  401238: 48 89 7d e8           mov    %rdi,-0x18(%rbp)
  40123c: c7 45 fc 00 00 00 00  movl   $0x0,-0x4(%rbp)
  401243: 48 83 7d e8 00        cmpq   $0x0,-0x18(%rbp)
  401248: 74 27                 je     401271
  40124a: 48 8b 45 e8           mov    -0x18(%rbp),%rax
  40124e: 0f b6 00              movzbl (%rax),%eax
  401251: 84 c0                 test   %al,%al
  401253: 74 1c                 je     401271
  401255: eb 0f                 jmp    401266
  401257: 48 8b 45 e8           mov    -0x18(%rbp),%rax
  40125b: 48 83 c0 01           add    $0x1,%rax
  40125f: 48 89 45 e8           mov    %rax,-0x18(%rbp)
  401263: 83 45 fc 01           addl   $0x1,-0x4(%rbp)
  401267: 48 8b 45 e8           mov    -0x18(%rbp),%rax
  40126b: 0f b6 00              movzbl (%rax),%eax
  40126e: 84 c0                 test   %al,%al
  401270: 75 e5                 jne    401257
  401272: 8b 45 fc              mov    -0x4(%rbp),%eax
  401275: c9                    leave
  401276: c3                    ret
  401277: 55                    push   %rbp
  401278: 48 89 e5              mov    %rsp,%rbp
  40127b: 48 83 ec 30           sub    $0x30,%rsp
  40127f: 48 89 7d d8           mov    %rdi,-0x28(%rbp)
  401283: c7 45 f8 00 00 00 00  movl   $0x0,-0x8(%rbp)
  40128a: c7 45 fc 00 00 00 00  movl   $0x0,-0x4(%rbp)
  401291: eb 36                 jmp    4012c9
  401293: 48 8b 45 d8           mov    -0x28(%rbp),%rax
  401297: 48 89 c7              mov    %rax,%rdi
  40129a: e8 91 ff ff ff        call   401230
  40129f: 3b 45 fc              cmp    -0x4(%rbp),%eax
  4012a2: 7e 21                 jle    4012c5
  4012a4: 8b 45 fc              mov    -0x4(%rbp),%eax
  4012a7: 48 98                 cltq
  4012a9: 48 8d 14 85 00 00 00  lea    0x0(,%rax,4),%rdx
  4012b0: 00
  4012b1: 48 8b 45 d8           mov    -0x28(%rbp),%rax
  4012b5: 48 01 d0              add    %rdx,%rax
  4012b8: 0f b6 00              movzbl (%rax),%eax
  4012bb: 0f be c0              movsbl %al,%eax
  4012be: 2d 30 00 00 00        sub    $0x30,%eax
  4012c3: 01 45 f8              add    %eax,-0x8(%rbp)
  4012c6: 83 45 fc 01           addl   $0x1,-0x4(%rbp)
  4012ca: 48 8b 45 d8           mov    -0x28(%rbp),%rax
  4012ce: 48 89 c7              mov    %rax,%rdi
  4012d1: e8 5a ff ff ff        call   401230
  4012d6: 39 45 fc              cmp    %eax,-0x4(%rbp)
  4012d9: 7c b8                 jl     401293
  4012db: 8b 45 f8              mov    -0x8(%rbp),%eax
  4012de: c9                    leave
  4012df: c3                    ret
  4012e0: 55                    push   %rbp
  4012e1: 48 89 e5              mov    %rsp,%rbp
  4012e4: 48 83 ec 10           sub    $0x10,%rsp
  4012e8: 89 7d fc              mov    %edi,-0x4(%rbp)
  4012eb: 89 75 f8              mov    %esi,-0x8(%rbp)
  4012ee: 83 7d fc 00           cmpl   $0x0,-0x4(%rbp)
  4012f2: 7e 0f                 jle    401303
  4012f4: 8b 45 fc              mov    -0x4(%rbp),%eax
  4012f7: 0f af 45 f8           imul   -0x8(%rbp),%eax
  4012fb: 89 c7                 mov    %eax,%edi
  4012fd: e8 0e 00 00 00        call   401310
  401302: 90                    nop
  401303: b8 00 00 00 00        mov    $0x0,%eax
  401308: c9                    leave
  401309: c3                    ret
  40130a: 66 0f 1f 44 00 00     nopw   0x0(%rax,%rax,1)
  401310: 55                    push   %rbp
  401311: 48 89 e5              mov    %rsp,%rbp
  401314: 48 83 ec 20           sub    $0x20,%rsp
  401318: 89 7d ec              mov    %edi,-0x14(%rbp)
  40131b: c7 45 fc 01 00 00 00  movl   $0x1,-0x4(%rbp)
  401322: c7 45 f8 01 00 00 00  movl   $0x1,-0x8(%rbp)
  401329: eb 0e                 jmp    401339
  40132b: 8b 45 fc              mov    -0x4(%rbp),%eax
  40132e: 0f af 45 f8           imul   -0x8(%rbp),%eax
  401332: 89 45 fc              mov    %eax,-0x4(%rbp)
  401335: 83 45 f8 01           addl   $0x1,-0x8(%rbp)
  401339: 8b 45 f8              mov    -0x8(%rbp),%eax
  40133c: 3b 45 ec              cmp    -0x14(%rbp),%eax
  40133f: 7e ea                 jle    40132b
  401341: 8b 45 fc              mov    -0x4(%rbp),%eax
  401344: c9                    leave
  401345: c3                    ret
  401346: 55                    push   %rbp
  401347: 48 89 e5              mov    %rsp,%rbp
  40134a: 48 83 ec 10           sub    $0x10,%rsp
  40134e: c7 45 fc 00 00 00 00  movl   $0x0,-0x4(%rbp)
  401355: c7 45 f8 0a 00 00 00  movl   $0xa,-0x8(%rbp)
  40135c: 8b 55 f8              mov    -0x8(%rbp),%edx
  40135f: 8b 45 fc              mov    -0x4(%rbp),%eax
  401362: 89 d6                 mov    %edx,%esi
  401364: 89 c7                 mov    %eax,%edi
  401366: e8 75 ff ff ff        call   4012e0
  40136b: b8 00 00 00 00        mov    $0x0,%eax
  401370: c9                    leave
  401371: c3                    ret
  401372: 66 2e 0f 1f 84 00 00  cs nopw 0x0(%rax,%rax,1)
  401379: 00 00 00
  40137c: 55                    push   %rbp
  40137d: 48 89 e5              mov    %rsp,%rbp
  401380: 48 83 ec 40           sub    $0x40,%rsp
  401384: 89 7d cc              mov    %edi,-0x34(%rbp)
  401387: 48 89 75 c0           mov    %rsi,-0x40(%rbp)
  40138b: 64 48 8b 04 25 28 00  mov    %fs:0x28,%rax
  401392: 00 00
  401394: 48 89 45 f8           mov    %rax,-0x8(%rbp)
  401398: 31 c0                 xor    %eax,%eax
  40139a: 48 8d 05 63 0c 00 00  lea    0xc63(%rip),%rax
  4013a1: 48 89 c7              mov    %rax,%rdi
  4013a4: e8 a7 fc ff ff        call   401050
  4013a9: 48 8d 45 d0           lea    -0x30(%rbp),%rax
  4013ad: 48 89 c7              mov    %rax,%rdi
  4013b0: e8 bb fc ff ff        call   401070
  4013b5: 48 8d 45 d0           lea    -0x30(%rbp),%rax
  4013b9: 48 89 c7              mov    %rax,%rdi
  4013bc: e8 0f fe ff ff        call   4011d0
  4013c1: 89 c6                 mov    %eax,%esi
  4013c3: 48 8d 05 4a 0c 00 00  lea    0xc4a(%rip),%rax
  4013ca: 48 89 c7              mov    %rax,%rdi
  4013cd: b8 00 00 00 00        mov    $0x0,%eax
  4013d2: e8 79 fc ff ff        call   401050
  4013d7: b8 00 00 00 00        mov    $0x0,%eax
  4013dc: 48 8b 55 f8           mov    -0x8(%rbp),%rdx
  4013e0: 64 48 2b 14 25 28 00  sub    %fs:0x28,%rdx
  4013e7: 00 00
  4013e9: 74 05                 je     4013f0
  4013eb: e8 60 fc ff ff        call   401050
  4013f0: c9                    leave
  4013f1: c3                    ret
  4013f2: 55                    push   %rbp
  4013f3: 48 89 e5              mov    %rsp,%rbp
  4013f6: 48 83 ec 10           sub    $0x10,%rsp
  4013fa: 48 89 7d f8           mov    %rdi,-0x8(%rbp)
  4013fe: 48 8b 45 f8           mov    -0x8(%rbp),%rax
  401402: 48 89 c7              mov    %rax,%rdi
  401405: e8 26 fe ff ff        call   401230
  40140a: 83 f8 05              cmp    $0x5,%eax
  40140d: 7e 16                 jle    401425
  40140f: 48 8b 45 f8           mov    -0x8(%rbp),%rax
  401413: 48 83 c0 05           add    $0x5,%rax
  401417: 0f b6 00              movzbl (%rax),%eax
  40141a: 3c 61                 cmp    $0x61,%al
  40141c: 75 07                 jne    401425
  40141e: b8 01 00 00 00        mov    $0x1,%eax
  401423: eb 05                 jmp    40142a
  401425: b8 00 00 00 00        mov    $0x0,%eax
  40142a: c9                    leave
  40142b: c3                    ret
  40142c: 55                    push   %rbp
  40142d: 48 89 e5              mov    %rsp,%rbp
  401430: 48 83 ec 20           sub    $0x20,%rsp
  401434: 89 7d ec              mov    %edi,-0x14(%rbp)
  401437: c7 45 fc 00 00 00 00  movl   $0x0,-0x4(%rbp)
  40143e: c7 45 f8 00 00 00 00  movl   $0x0,-0x8(%rbp)
  401445: eb 09                 jmp    401450
  401447: 8b 45 fc              mov    -0x4(%rbp),%eax
  40144a: 01 45 f8              add    %eax,-0x8(%rbp)
  40144d: 83 45 fc 01           addl   $0x1,-0x4(%rbp)
  401451: 8b 45 fc              mov    -0x4(%rbp),%eax
  401454: 3b 45 ec              cmp    -0x14(%rbp),%eax
  401457: 7c ee                 jl     401447
  401459: 8b 45 f8              mov    -0x8(%rbp),%eax
  40145c: c9                    leave
  40145d: c3                    ret
  40145e: 66 90                 xchg   %ax,%ax
  401460: 41 57                 push   %r15
  401462: 4c 8d 3d 07 29 00 00  lea    0x2907(%rip),%r15
  401469: 41 56                 push   %r14
  40146b: 49 89 d6              mov    %rdx,%r14
  40146e: 41 55                 push   %r13
  401470: 49 89 f5              mov    %rsi,%r13
  401473: 41 54                 push   %r12
  401475: 41 89 fc              mov    %edi,%r12d
  401478: 55                    push   %rbp
  401479: 48 8d 2d f8 28 00 00  lea    0x28f8(%rip),%rbp
  401480: 53                    push   %rbx
  401481: 4c 29 fd              sub    %r15,%rbp
  401484: 48 83 ec 08           sub    $0x8,%rsp
  401488: e8 73 fb ff ff        call   401000
  40148d: 48 c1 fd 03           sar    $0x3,%rbp
  401491: 74 1f                 je     4014b2
  401493: 31 db                 xor    %ebx,%ebx
  401495: 0f 1f 00              nopl   (%rax)
  401498: 4c 89 f2              mov    %r14,%rdx
  40149b: 4c 89 ee              mov    %r13,%rsi
  40149e: 44 89 e7              mov    %r12d,%edi
  4014a1: 41 ff 14 df           call   *(%r15,%rbx,8)
  4014a5: 48 83 c3 01           add    $0x1,%rbx
  4014a9: 48 39 dd              cmp    %rbx,%rbp
  4014ac: 75 ea                 jne    401498
  4014ae: 48 83 c4 08           add    $0x8,%rsp
  4014b2: 5b                    pop    %rbx
  4014b3: 5d                    pop    %rbp
  4014b4: 41 5c                 pop    %r12
  4014b6: 41 5d                 pop    %r13
  4014b8: 41 5e                 pop    %r14
  4014ba: 41 5f                 pop    %r15
  4014bc